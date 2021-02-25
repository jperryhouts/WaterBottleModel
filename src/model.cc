// Water Bottle Model for lower crustal flow.
// Copyright (C) 2020 Jonathan Perry-Houts
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include "model.h"

template <int spacedim>
CrustalFlow<spacedim>::CrustalFlow (const MPI_Comm mpi_communicator_)
  : MPI_COMM (Utilities::MPI::duplicate_communicator (mpi_communicator_)),
    iostream_tee_device(std::cout, log_file_stream),
    iostream_tee_stream(iostream_tee_device),
    pcout (iostream_tee_stream,
          (Utilities::MPI::this_mpi_process(MPI_COMM) == 0)),
    computing_timer (MPI_COMM, pcout, TimerOutput::summary,
        TimerOutput::wall_times),
    output_directory(""),
    surface_mesh (MPI_COMM,
                      typename Triangulation<dim, spacedim>::MeshSmoothing (
                        Triangulation<dim, spacedim>::smoothing_on_refinement
                        | Triangulation<dim, spacedim>::smoothing_on_coarsening),
                      parallel::distributed::Triangulation<dim, spacedim>::mesh_reconstruction_after_repartitioning),
    mapping(flexure_degree), /* Use mapping degree consistent with equations, to accurately represent surface deformations. */
    dof_handler (surface_mesh),
    fe (FE_Q<dim, spacedim> (velocity_degree), spacedim /* Crustal flow velocity */,
    FE_Q<dim, spacedim> (h_degree), 1 /* h -- Lower crustal thickness */,
    FE_Q<dim, spacedim> (flexure_degree-1), 1 /* v -- Elastic plate laplacian */,
    FE_Q<dim, spacedim> (flexure_degree), 1 /* w -- Elastic plate displacement */,
    FE_Q<dim, spacedim> (s_degree), 1 /* s -- Sill thickness */),
    u_extractor (u_component_index),
    h_extractor (h_component_index),
    v_extractor (v_component_index),
    w_extractor (w_component_index),
    s_extractor (s_component_index),
    // All of the following are actually scalar fields, but the ones
    // which are to be applied as boundary conditions need to have
    // as many components as there are FE fields (even though all but)
    // one will be masked when the function is actually evaluated).
    initial_crustal_thickness_field (spacedim+4),
    topographic_boundary_value_field (spacedim+4),
    prescribed_overburden_field (1),
    rigidity_field (1),
    viscosity_field (1),
    initial_sill_thickness_field (spacedim+4),
    sill_emplacement_field (1)
  {}

template <int spacedim>
CrustalFlow<spacedim>::~CrustalFlow ()
  {
    dof_handler.clear();

    // wait if there is a thread that's still writing the statistics
    // object (set from the output_statistics() function)
    output_statistics_thread.join();

    // If an exception is being thrown (for example due to AssertThrow()), we
    // might end up here with currently active timing sections. The destructor
    // of TimerOutput does MPI communication, which can lead to deadlocks,
    // hangs, or confusing MPI error messages. To avoid this, we can call
    // reset() to remove all open sections. In a normal run, we won't have any
    // active sessions, so this won't hurt to do:
    computing_timer.reset();
  }

template <int spacedim>
void
CrustalFlow<spacedim>::
output_statistics()
{
  if (Utilities::MPI::this_mpi_process(MPI_COMM)!=0)
      return;

  // Formatting the table we're about to output and writing the
  // actual file may take some time, so do it on a separate
  // thread. We do this using a lambda function that takes
  // a copy of the statistics object to make sure that whatever
  // we do to the 'real' statistics object at the time of
  // writing data doesn't affect what we write.
  //
  // Before we can start working on a new thread, we need to
  // make sure that the previous thread is done or they'll
  // step on each other's feet.
  output_statistics_thread.join();

  // TODO[C++14]: The following code could be made significantly simpler
  // if we could just copy the statistics table as part of the capture
  // list of the lambda function. In C++14, this would then simply be
  // written as
  //   [statistics_copy = this->statistics, this] () {...}
  // (It would also be nice if we could use a std::unique_ptr, but since
  // these can not be copied and since lambda captures don't allow move
  // syntax for captured values, this also doesn't work. This can be done
  // in C++14 by writing
  //   [statistics_copy_ptr = std::move(statistics_copy_ptr), this] () {...}
  // but, as mentioned above, if we could use C++14, we wouldn't have to
  // use a pointer in the first place.)
  std::shared_ptr<TableHandler> statistics_copy_ptr
    = std_cxx14::make_unique<TableHandler>(statistics);
  auto write_statistics
    = [statistics_copy_ptr,this]()
  {
    // First write everything into a string in memory
    std::ostringstream stream;
    statistics_copy_ptr->write_text (stream,
                                     TableHandler::table_with_separate_column_description);
    stream.flush();

    const std::string statistics_contents = stream.str();

    // Next find out whether we need to write everything into
    // the statistics file, or whether it is enough to just write
    // the last few bytes that were added since we wrote to that
    // file again. The way we do that is by checking whether the
    // first few bytes of the string we just created match what we
    // had previously written. One might think that they always should,
    // but the statistics object automatically sizes the column widths
    // of its output to match what is being written, and so if a later
    // entry requires more width, then even the first columns are
    // changed -- in that case, we will have to write everything,
    // not just append one line.
    const bool write_everything
      = ( // We may have never written anything. More precisely, this
          // case happens if the statistics_last_write_size is at the
          // value initialized by the Simulator::Simulator()
          // constructor, and this can happen in two situations:
          // (i) At the end of the first time step; and (ii) upon restart
          // since the variable we query here is not serialized. It is clear
          // that in both situations, we want to write the
          // entire contents of the statistics object. For the second
          // case, this is also appropriate since someone may have
          // previously restarted from a checkpoint, run a couple of
          // time steps that have added to the statistics file, but then
          // aborted the run again; a later restart from the same
          // checkpoint then requires overwriting the statistics file
          // on disk with what we have when this function is called for
          // the first time after the restart. The same situation
          // happens if the simulation kept running for some time after
          // a checkpoint, but is resumed from that checkpoint (i.e.,
          // at an earlier time step than when the statistics file was
          // written to last). In these situations, we effectively want
          // to "truncate" the file to the state stored in the checkpoint,
          // and we do that by just overwriting the entire file.
          (statistics_last_write_size == 0)
          ||
          // Or the size of the statistics file may have
          // shrunk mysteriously -- this shouldn't happen
          // but if it did we'd get into trouble with the
          // .substr() call in the next check.
          (statistics_last_write_size > statistics_contents.size())
          ||
          // Or the hash of what we wrote last time doesn't match
          // the hash of the first part of what we want to write
          (statistics_last_hash
           !=
           std::hash<std::string>()(statistics_contents.substr(0, statistics_last_write_size))) );

    const std::string stat_file_name = output_directory + "statistics";
    if (write_everything)
      {
        // Write what we have into a tmp file, then move that into
        // place
        const std::string tmp_file_name = stat_file_name + ".tmp";
        {
          std::ofstream tmp_file (tmp_file_name);
          tmp_file << statistics_contents;
        }
        std::rename(tmp_file_name.c_str(), stat_file_name.c_str());
      }
    else
      {
        // If we don't have to write everything, then the first part of what
        // we want to write matches what's already on disk. In that case,
        // we just have to append what's new.
        std::ofstream stat_file (stat_file_name, std::ios::app);
        stat_file << statistics_contents.substr(statistics_last_write_size, std::string::npos);
      }

    // Now update the size and hash of what we just wrote so that
    // we can compare against it next time we get here. Note that we do
    // not need to guard access to these variables with a mutex because
    // this is the only function that touches the variables, and
    // this function runs only once at a time (on a different
    // thread, but it's not started a second time while the previous
    // run hasn't finished).
    statistics_last_write_size = statistics_contents.size();
    statistics_last_hash       = std::hash<std::string>()(statistics_contents);
  };
  output_statistics_thread = Threads::new_thread (write_statistics);
}

template <int spacedim>
void
CrustalFlow<spacedim>::
set_outputdir (const std::string &dirname)
  {
    output_directory = dirname;
    if (output_directory.size() == 0)
      output_directory = "./";
    else if (output_directory[output_directory.size()-1] != '/')
      output_directory += "/";
  }

template <int spacedim>
void
CrustalFlow<spacedim>::
setup_dofs()
  {
    TimerOutput::Scope timing_section (computing_timer, "Setting up dofs");

    dof_handler.distribute_dofs (fe);
    //DoFRenumbering::Cuthill_McKee(dof_handler);
    DoFRenumbering::Cuthill_McKee(dof_handler,false,true);
    {
      locally_owned_dofs = dof_handler.locally_owned_dofs ();
      DoFTools::extract_locally_relevant_dofs (dof_handler,
                                               locally_relevant_dofs);
    }

    {
      constraints.clear ();
      constraints.reinit (locally_relevant_dofs);
      DoFTools::make_hanging_node_constraints (dof_handler, constraints);

      VectorTools::interpolate_boundary_values (dof_handler, 0, initial_crustal_thickness_field,
                                                // Functions::ConstantFunction<spacedim>(
                                                //    this->crustal_thickness_boundary_value, spacedim+4),
                                                constraints,
                                                fe.component_mask(h_extractor));
      VectorTools::interpolate_boundary_values (dof_handler, 0, this->topographic_boundary_value_field,
                                                constraints,
                                                fe.component_mask(w_extractor));
      VectorTools::interpolate_boundary_values (dof_handler, 0, this->initial_sill_thickness_field,
                                                constraints,
                                                fe.component_mask(s_extractor));
      VectorTools::interpolate_boundary_values (dof_handler, 0,
                                                Functions::ZeroFunction<spacedim>(spacedim+4),
                                                constraints,
                                                fe.component_mask(v_extractor));

      if (spacedim == 2)
        { // Need to apply boundary conditions to both left and right edges if we're working in dim=1 (spacedim=2)
          // In spacedim=3 all edges are treated with the same boundary_id.
          VectorTools::interpolate_boundary_values (dof_handler, 1, this->initial_crustal_thickness_field,
                                                    constraints,
                                                    fe.component_mask(h_extractor));
          VectorTools::interpolate_boundary_values (dof_handler, 1, this->topographic_boundary_value_field,
                                                    constraints,
                                                    fe.component_mask(w_extractor));
          VectorTools::interpolate_boundary_values (dof_handler, 1, this->initial_sill_thickness_field,
                                                    constraints,
                                                    fe.component_mask(s_extractor));
          VectorTools::interpolate_boundary_values (dof_handler, 1,
                                                    Functions::ZeroFunction<spacedim>(spacedim+4),
                                                    constraints,
                                                    fe.component_mask(v_extractor));
        }

      std::vector<Point<spacedim>> constraint_locations;
      std::vector<unsigned int>    constraint_component_indices;
      std::vector<double>          constraint_values;

      // { // Handle elastic plate displacement nullspace
      //   Point<spacedim> pt;
      //   // pt[0] = 0.5;
      //   // pt[1] = 0.5;
      //   constraint_locations.push_back(pt);
      //   constraint_component_indices.push_back(h_component_index);
      //   constraint_values.push_back(h_0);
      // }

      for (unsigned int i=0; i<constraint_locations.size(); ++i)
      { // BEGIN CONSTRAIN A POINT
        const Point<spacedim> constraint_location = constraint_locations[i];
        const unsigned int constraint_component_index = constraint_component_indices[i];
        const double constraint_value = constraint_values[i];

        double min_local_distance = std::numeric_limits<double>::max();
        types::global_dof_index local_best_dof_index;
        Point<spacedim> nearest_local_point;

        const std::vector<Point<dim>> points = fe.get_unit_support_points();
        const Quadrature<dim> quadrature(points);
        FEValues<dim,spacedim> fe_values(mapping, fe, quadrature, update_quadrature_points);
        std::vector<unsigned int> local_dof_indices(fe.dofs_per_cell);

        for (const auto &cell : dof_handler.active_cell_iterators())
          if (!cell->is_artificial()) //(cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            cell->get_dof_indices(local_dof_indices);

            //for (unsigned int q=0;q<fe.dofs_per_cell; ++q)
            for (unsigned int q=0; q<quadrature.size(); ++q)
              {
                const unsigned int component_idx = fe.system_to_component_index(q).first;
                if (component_idx == constraint_component_index)
                  {
                    const types::global_dof_index idx = local_dof_indices[q];
                    if (constraints.can_store_line(idx) &&
                        !constraints.is_constrained(idx))
                      {
                        Point<spacedim> p = fe_values.quadrature_point(q);
                        const double distance = constraint_location.distance(p);
                        if (distance < min_local_distance)
                        {
                          nearest_local_point = p;
                          min_local_distance = distance;
                          local_best_dof_index = idx;
                        }
                      }
                  }
              }
          }

        const double global_nearest = Utilities::MPI::min(min_local_distance,
                                                          MPI_COMM);
        if (std::abs(min_local_distance-global_nearest) < 1e-12)
        { // Directly comparing floating point numbers freaks me out.
          if (constraints.can_store_line(local_best_dof_index))
          {
            constraints.add_line(local_best_dof_index);
            constraints.set_inhomogeneity(local_best_dof_index, constraint_value);

            // pcout << "Found a point to constrain on MPI process " <<
            //   Utilities::MPI::this_mpi_process(MPI_COMM) <<
            //   " at location (" << nearest_local_point[0] << ", " << nearest_local_point[1];
            // if (spacedim == 3)
            //   pcout << ", " << nearest_local_point[2];
            // pcout << ")" << std::endl;
          }
        }
      } // END CONSTRAIN A POINT

      constraints.close ();
    }

    {
      DynamicSparsityPattern dsp(locally_relevant_dofs);
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
      SparsityTools::distribute_sparsity_pattern(dsp,
                                                 dof_handler.locally_owned_dofs(),
                                                 MPI_COMM,
                                                 locally_relevant_dofs);
      system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp,
                           MPI_COMM);

      rhs.reinit (locally_owned_dofs, MPI_COMM);
      locally_relevant_solution.reinit (locally_owned_dofs, locally_relevant_dofs, MPI_COMM);
      old_locally_relevant_solution.reinit (locally_owned_dofs, locally_relevant_dofs, MPI_COMM);
      tmp_locally_relevant_solution.reinit (locally_owned_dofs, locally_relevant_dofs, MPI_COMM);
    }
  }

template <int spacedim>
void
CrustalFlow<spacedim>::
assemble_system(const double dt)
  {
    TimerOutput::Scope timing_section (computing_timer, "Assembling system matrices");
    pcout << "  Assembling system..." << std::flush;

    system_matrix = 0;
    rhs = 0;

    const QGauss<dim> quadrature_formula (flexure_degree+1);
    FEValues<dim,spacedim> fe_values (mapping, fe, quadrature_formula,
                                      update_values | update_gradients
                                      | update_normal_vectors
                                      | update_quadrature_points
                                      | update_JxW_values);
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size ();
    FullMatrix<double> cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    std::vector<Tensor<1,spacedim>> phi_u (dofs_per_cell);
    std::vector<double> phi_v (dofs_per_cell);
    std::vector<double> phi_w (dofs_per_cell);
    std::vector<double> phi_h (dofs_per_cell);
    std::vector<double> phi_s (dofs_per_cell);

    std::vector<Tensor<1,spacedim>> grad_phi_v (dofs_per_cell);
    std::vector<Tensor<1,spacedim>> grad_phi_w (dofs_per_cell);
    std::vector<Tensor<1,spacedim>> grad_phi_h (dofs_per_cell);
    std::vector<double> div_phi_u (dofs_per_cell);
    std::vector<SymmetricTensor<2, spacedim>> grad_phi_u (dofs_per_cell);

    std::vector<double> current_h_values (n_q_points);
    std::vector<Tensor<1,spacedim>> current_w_gradients (n_q_points);

    std::vector<double> old_h_values (n_q_points);
    std::vector<double> old_s_values (n_q_points);

    Point<spacedim> loc;
    double D, eta, h_bar, h_n, s_n, sigma, emplacement = 0;
    Tensor<1,spacedim> grad_w_bar;

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell_matrix = 0;
          cell_rhs = 0;
          fe_values.reinit (cell);

          fe_values[h_extractor].get_function_values
            (locally_relevant_solution, current_h_values);
          fe_values[w_extractor].get_function_gradients
            (locally_relevant_solution, current_w_gradients);

          fe_values[h_extractor].get_function_values
            (old_locally_relevant_solution, old_h_values);
          fe_values[s_extractor].get_function_values
            (old_locally_relevant_solution, old_s_values);

          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                  phi_u[k] = fe_values[u_extractor].value (k,q);
                  phi_v[k] = fe_values[v_extractor].value (k,q);
                  phi_w[k] = fe_values[w_extractor].value (k,q);
                  phi_h[k] = fe_values[h_extractor].value (k,q);
                  phi_s[k] = fe_values[s_extractor].value (k,q);
                  div_phi_u[k] = fe_values[u_extractor].divergence (k,q);
                  grad_phi_u[k] = fe_values[u_extractor].symmetric_gradient (k,q);
                  grad_phi_h[k] = fe_values[h_extractor].gradient (k,q);
                  grad_phi_v[k] = fe_values[v_extractor].gradient (k,q);
                  grad_phi_w[k] = fe_values[w_extractor].gradient (k,q);
                }

              loc = fe_values.quadrature_point (q);

              D = rigidity_field.value(loc);
              eta = viscosity_field.value(loc);

              if (initialization_step || crustal_flow_time == 0)
                {
                  h_bar = h_n = initial_crustal_thickness_field.value(loc);
                  s_n = initial_sill_thickness_field.value(loc);
                  grad_w_bar *= 0;
                }

              if (!initialization_step)
                {
                  h_bar = current_h_values[q];
                  grad_w_bar = current_w_gradients[q];
                }

              if (crustal_flow_time > 0)
                {
                  h_n = old_h_values[q];
                  s_n = old_s_values[q];
                }

              s_n = std::max(0.0, s_n);

              if (this->use_prescribed_overburden)
                {
                  sigma = prescribed_overburden_field.value(loc);
                }
              // else if (unit_testing)
              //   { const double omega = test_perturbation_freq*dealii::numbers::PI/model_width;
              //     const double X = loc[0];
              //     sigma = -(0.7 + (0.1*(std::pow(omega,4.0) + 1)*std::sin(omega*X)));
              //     //grad_w_bar[0] = 0.1*omega*std::cos(omega*X);
              //     eta = 1.0; D = 1.0; rho_c = 0.85; rho_m = 1.0; gravity = 1.0; }
              else
                {
                  sigma = (rho_s-rho_c)*gravity*s_n;
                  emplacement = sill_emplacement_field.value(loc);
                }

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      cell_matrix (i, j) += (//  eta grad^2 U - 2*eta/(gamma*h)^2
                                             // -eta*(grad_phi_u[i],grad_phi_u[j]) - (...)*phi_u[i]*phi_u[j]
                                              - eta * h_bar * scalar_product(grad_phi_u[i], grad_phi_u[j])
                                                - ( 2.0 * eta / (gamma * h_bar)
                                                    * scalar_product(phi_u[i], phi_u[j]) )

                                                + ((dt > 0) ? 1.0 : 0.0) *
                                                  2.0 * Pi * (rho_c - rho_m) * gravity * h_bar
                                                  * scalar_product(phi_u[i], grad_phi_h[j])
                                              //// h + dt*2/3*div_u*h_bar = h_n
                                              //// + phi_h[i] * h_bar * dt * (2.0/3.0) * div_phi_u[j]
                                              //// + phi_h[i] * phi_h[j]
                                              + ((dt > 0) ?
                                                (phi_h[i] * div_phi_u[j] *
                                                  2.0 * Pi * (rho_c - rho_m) * gravity * h_bar
                                                // - 3.0 * Pi * (rho_c - rho_m) * gravity * h_n / (dt * /* phi_h[j] */ gamma)
                                                + phi_h[i] * phi_h[j]
                                                  * 3.0 * Pi * (rho_c - rho_m) * gravity / (gamma * dt))
                                                :
                                                phi_h[i] * phi_h[j])

                                              // // Artificial diffusion
                                              //+ 1e-2 * dt * scalar_product(grad_phi_h[i], grad_phi_h[j])

                                              // D/gamma grad^2 v + rho_m*g*w =>
                                              //-D/gamma (grad phi_v_i, grad_phi_v_j) + rho_m*g*phi_v_i*phi_w_j
                                              - (gamma*D)/(rho_m*gravity) * scalar_product(grad_phi_v[i], grad_phi_v[j])
                                              + phi_v[i] * phi_w[j]

                                              // v - grad^2 w = 0
                                              // phi_w[i]*phi_v[j] + (grad_phi_w[i],grad_phi_w[j])
                                              + phi_w[i] * phi_v[j]
                                              + scalar_product(grad_phi_w[i], grad_phi_w[j])

                                              + phi_s[i] * phi_s[j]
                                            ) * fe_values.JxW (q) ;
                    }
                  cell_rhs (i) += (// phi_u[i] * (-Pi * grad P)
                                    Pi  * (rho_c-rho_m) * gravity * h_bar
                                        * scalar_product(phi_u[i], grad_w_bar)

                                    // phi_h[i] * h_bar
                                    + phi_h[i] * ((dt > 0) ?
                                      3.0 * Pi * (rho_c - rho_m) * gravity * h_n / (gamma * dt)
                                      //Pi * 3.0 * (rho_m - rho_c) * gravity / (/*dt */ gamma)
                                      :
                                      h_n)

                                    // phi_v[i] * ((rhom-rhos)*g*S + 2*(rhom-rhoc)*g*h)
                                    + phi_v[i]
                                      * (2 * (rho_m - rho_c) * h_bar / rho_m
                                        - sigma / (rho_m * gravity))

                                    // 0 * phi_w[i]

                                    // phi_s[i] * (s_n + dt * ds/dt)
                                    + phi_s[i] * std::max(0.0, s_n + dt*emplacement)
                                  ) * fe_values.JxW (q);
                }
            }
          cell->get_dof_indices (local_dof_indices);
          constraints.distribute_local_to_global (cell_matrix, cell_rhs,
                                                  local_dof_indices,
                                                  system_matrix, rhs);
        }
    system_matrix.compress (VectorOperation::add);
    rhs.compress (VectorOperation::add);

    pcout << " OK" << std::endl;
  }

template <int spacedim>
std::pair<double, double>
CrustalFlow<spacedim>::
picard_residuals (const TrilinosWrappers::MPI::Vector &old_solution,
                  const TrilinosWrappers::MPI::Vector &new_solution)
  {
    const QGauss<dim> quadrature_formula (this->flexure_degree+1);
    FEValues<dim,spacedim> fe_values (mapping, fe, quadrature_formula,
                                      update_values | update_JxW_values);
    const unsigned int n_q_points = quadrature_formula.size();

    std::vector<double> previous_h_values(n_q_points);
    std::vector<double> current_h_values(n_q_points);
    std::vector<double> previous_w_values(n_q_points);
    std::vector<double> current_w_values(n_q_points);

    double local_h_residual_integral = 0.0;
    double local_w_residual_integral = 0.0;

    // compute the integral quantities by quadrature
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit (cell);
          fe_values[this->h_extractor].get_function_values (old_solution, previous_h_values);
          fe_values[this->h_extractor].get_function_values (new_solution, current_h_values);

          fe_values[this->w_extractor].get_function_values (old_solution, previous_w_values);
          fe_values[this->w_extractor].get_function_values (new_solution, current_w_values);

          double w_q_residual, h_q_residual;
          for (unsigned int q=0; q<n_q_points; ++q)
            {
              h_q_residual = std::pow(current_h_values[q] - previous_h_values[q], 2);
              local_h_residual_integral += h_q_residual*fe_values.JxW(q);

              w_q_residual = std::pow(current_w_values[q] - previous_w_values[q], 2);
              local_w_residual_integral += w_q_residual*fe_values.JxW(q);
            }
        }

    // compute the sum over all processors
    const double global_h_residual
      = std::sqrt(Utilities::MPI::sum (local_h_residual_integral, MPI_COMM));
    const double global_w_residual
      = std::sqrt(Utilities::MPI::sum (local_w_residual_integral, MPI_COMM));

    return std::make_pair(global_h_residual, global_w_residual);
  }

template <int spacedim>
std::pair<double,double>
CrustalFlow<spacedim>::
solve_direct ()
  {
    TimerOutput::Scope timing_section (computing_timer, "Direct solver");
    pcout << "  Solving direct..." << std::flush;

    TrilinosWrappers::MPI::Vector fully_distributed_solution (locally_owned_dofs, MPI_COMM);

    SolverControl sc;
    TrilinosWrappers::SolverDirect direct_solver (sc);
    try
      {
        direct_solver.solve (system_matrix, fully_distributed_solution, rhs);
        constraints.distribute (fully_distributed_solution);

        tmp_locally_relevant_solution = fully_distributed_solution;
        const std::pair<double, double> residuals
          = picard_residuals(locally_relevant_solution,
                             tmp_locally_relevant_solution);
        const double h_residual = residuals.first,
                     w_residual = residuals.second;

        const double residual_norm = (h_residual+w_residual)/(std::pow(model_width,spacedim-1));

        if (residual_norm <= picard_tolerance)
          old_locally_relevant_solution = locally_relevant_solution;

        locally_relevant_solution = fully_distributed_solution;
        pcout << " OK" << std::endl;
        return residuals;
      }
    catch (const std::exception &exc)
      {
        if (Utilities::MPI::this_mpi_process (MPI_COMM) == 0)
          {
            AssertThrow(false, ExcMessage (
                        std::string ("The flow solver failed with error:\n\n")
                      + exc.what ()));
          }
        else
          {
            throw QuietException();
          }
      }
      pcout << " FAILED" << std::endl;
      return std::make_pair(-1, -1);
  }

template <int spacedim>
std::pair<double,double>
CrustalFlow<spacedim>::
solve_iterative ()
  {
    TimerOutput::Scope timing_section (computing_timer, "Solving system");
    pcout << "  Solving system..." << std::flush;

    // pcout << "Solving with CG" << std::endl;
    TrilinosWrappers::MPI::Vector fully_distributed_solution (locally_owned_dofs, MPI_COMM);

    const double tolerance = solver_relative_tolerance * rhs.l2_norm();
    SolverControl sc (dof_handler.n_dofs(), tolerance);
    SolverCG<TrilinosWrappers::MPI::Vector> iterative_solver (sc);

    TrilinosWrappers::PreconditionILU preconditioner;
    TrilinosWrappers::PreconditionILU::AdditionalData data;

    try
      {
        preconditioner.initialize(system_matrix, data);
        iterative_solver.solve(system_matrix, fully_distributed_solution,
                               rhs, preconditioner);
        constraints.distribute (fully_distributed_solution);

        // distributed_solution.compress(VectorOperation::add);

        tmp_locally_relevant_solution = fully_distributed_solution;
        const std::pair<double, double> residuals
          = picard_residuals(locally_relevant_solution,
                             tmp_locally_relevant_solution);
        const double h_residual = residuals.first,
                     w_residual = residuals.second;

        const double residual_norm = (h_residual+w_residual)/(std::pow(model_width,spacedim-1));

        if (residual_norm <= picard_tolerance)
          old_locally_relevant_solution = locally_relevant_solution;

        locally_relevant_solution = fully_distributed_solution;
        pcout << " OK" << std::endl;
        return residuals;
      }
    catch (const std::exception &exc)
      {
        if (Utilities::MPI::this_mpi_process (MPI_COMM) == 0)
          pcout << "Iterative solver failed with error: " << exc.what()
                    << "\nTolerance: " << tolerance
                    << "\nFalling back to direct solver" << std::endl;
      }

    pcout << " FAILED" << std::endl;
    return solve_direct();
  }

template <int spacedim>
double
CrustalFlow<spacedim>::
get_dt(const double max_dt)
  {
    const QIterated<dim> quadrature_formula (QTrapez<1> (), velocity_degree);
    const unsigned int n_q_points = quadrature_formula.size ();
    FEValues<dim,spacedim> fe_values (mapping, fe, quadrature_formula, update_values | update_gradients);
    std::vector<Tensor<1, spacedim> > velocity_values (n_q_points);
    std::vector<double> div_u (n_q_points);
    std::vector<double> h_values (n_q_points);
    double max_local_cfl_cond = 0, max_local_rate, local_u, local_dhdt;

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned ())
        {
          fe_values.reinit (cell);
          fe_values[u_extractor].get_function_values (locally_relevant_solution, velocity_values);
          fe_values[u_extractor].get_function_divergences (locally_relevant_solution, div_u);
          fe_values[h_extractor].get_function_values (locally_relevant_solution, h_values);
          max_local_rate = 1e-10;
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              local_u = velocity_values[q].norm();
              local_dhdt = abs(2/3*div_u[q]*h_values[q]);
              max_local_rate = std::max(std::max(local_u, local_dhdt),
                                        max_local_rate);
            }
          max_local_cfl_cond = std::max(max_local_cfl_cond,
                                        max_local_rate/cell->diameter());
        }
    const double CFL_cond = Utilities::MPI::max (max_local_cfl_cond, MPI_COMM);

    return std::min (max_dt, CFL / CFL_cond);
  }

template <int spacedim>
void
CrustalFlow<spacedim>::
refine_mesh()
  {
    pcout << "  Refining mesh..." << std::flush;

    parallel::distributed::SolutionTransfer<dim,
             TrilinosWrappers::MPI::Vector, DoFHandler<dim,spacedim>>
             solutionTx (dof_handler);

    {
      Vector<float> estimated_error_per_cell (surface_mesh.n_active_cells ());
      KellyErrorEstimator<dim, spacedim>::estimate (dof_handler, QGauss<dim-1> (flexure_degree+1),
                                                    std::map<types::boundary_id, const Function<spacedim> *>(),
                                                    locally_relevant_solution,
                                                    estimated_error_per_cell,
                                                    fe.component_mask(h_extractor)
                                                    | fe.component_mask(s_extractor),
                                                    nullptr, 0, surface_mesh.locally_owned_subdomain ());
      GridRefinement::refine_and_coarsen_fixed_fraction (
        surface_mesh, estimated_error_per_cell, 0.5, 0.3);

      // Limit refinement to min/max levels
      if (surface_mesh.n_levels () > max_refinement)
        for (typename Triangulation<dim, spacedim>::active_cell_iterator cell =
               surface_mesh.begin_active ( max_refinement );
             cell != surface_mesh.end (); ++cell)
          cell->clear_refine_flag ();
      for (typename Triangulation<dim, spacedim>::active_cell_iterator cell =
             surface_mesh.begin_active ( min_refinement );
           cell != surface_mesh.end_active ( min_refinement ); ++cell)
        cell->clear_coarsen_flag ();

      // Transfer solution onto new mesh
      std::vector<const TrilinosWrappers::MPI::Vector *> solution (2);
      solution[0] = &locally_relevant_solution;
      solution[1] = &old_locally_relevant_solution;
      surface_mesh.prepare_coarsening_and_refinement ();
      solutionTx.prepare_for_coarsening_and_refinement (solution);
      surface_mesh.execute_coarsening_and_refinement ();
    }

    setup_dofs ();

    {
      TrilinosWrappers::MPI::Vector distributed_solution (rhs);
      TrilinosWrappers::MPI::Vector old_distributed_solution (rhs);
      std::vector<TrilinosWrappers::MPI::Vector *> tmp (2);
      tmp[0] = &(distributed_solution);
      tmp[1] = &(old_distributed_solution);
      solutionTx.interpolate (tmp);
      constraints.distribute (distributed_solution);
      constraints.distribute (old_distributed_solution);
      locally_relevant_solution = distributed_solution;
      old_locally_relevant_solution = old_distributed_solution;
    }

    pcout << " OK" << std::endl;
  }

template <int spacedim>
void
CrustalFlow<spacedim>::
output_results (const unsigned int timestep,
                const double time)
  {
    const std::string vis_directory = output_directory + "/crustal_flow";

    if (output_format == "vtu") {
      DataOut<dim,DoFHandler<dim,spacedim>> data_out;
      data_out.attach_dof_handler (dof_handler);

      std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(0);
      std::vector<std::string> solution_name(0);
      for (unsigned int i=0; i<spacedim; ++i)
        {
          data_component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
          solution_name.push_back("Velocity");
        }
      data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
      data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
      data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
      data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

      solution_name.push_back ("Lower_crust_half_thickness");
      solution_name.push_back ("Plate_curvature");
      solution_name.push_back ("Plate_displacement");
      solution_name.push_back ("Sill_thickness");

      data_out.add_data_vector (locally_relevant_solution, solution_name,
                                DataOut<dim,DoFHandler<dim,spacedim>>::type_dof_data,
                                data_component_interpretation);

      const Postprocessor::StaticFunctionPostprocessor<spacedim>
        overburden_vis ("Overburden", &(this->prescribed_overburden_field));

      if (this->use_prescribed_overburden)
        data_out.add_data_vector (locally_relevant_solution, overburden_vis);

      data_out.build_patches (mapping, mapping.get_degree());
      std::ofstream output (
        (vis_directory + "/crustal_flow-"
         + Utilities::int_to_string (timestep, 5) + "."
         + Utilities::int_to_string (surface_mesh.locally_owned_subdomain (), 4)
         + ".vtu").c_str ());
      data_out.write_vtu (output);

      if (Utilities::MPI::this_mpi_process(MPI_COMM) == 0)
        {
          std::vector<std::string> filenames;
          const unsigned int n_procs = Utilities::MPI::n_mpi_processes(MPI_COMM);
          for (unsigned int i=0; i<n_procs; ++i)
            filenames.push_back("crustal_flow-"
                              + Utilities::int_to_string (timestep, 5) + "."
                              + Utilities::int_to_string (i, 4) + ".vtu");

          const std::string pvtu_master_filename = "crustal_flow/crustal_flow-"
                                                 + Utilities::int_to_string (timestep, 4)
                                                 + ".pvtu";

          std::ofstream pvtu_master (output_directory + "/" + pvtu_master_filename);
          data_out.write_pvtu_record (pvtu_master, filenames);

          static std::vector<std::pair<double, std::string>> times_and_names;
          times_and_names.push_back (std::pair<double, std::string> (time, pvtu_master_filename));
          std::ofstream pvd_output (output_directory + "/crustal_flow.pvd");
          DataOutBase::write_pvd_record (pvd_output, times_and_names);
        }
    }
    else if (output_format == "ascii")
    {
      std::ofstream output;

      // const unsigned int this_mpi_process = Utilities::MPI::this_mpi_process(MPI_COMM);
      // if (this_mpi_process == 0)
      {
        std::string filename = vis_directory + "/crustal_flow-"
                             + Utilities::int_to_string(timestep, 4) + "."
                             + Utilities::int_to_string (surface_mesh.locally_owned_subdomain (), 4)
                             //+ Utilities::int_to_string(this_mpi_process, 4)
                             + ".txt";
        output.open(filename.c_str());

        output.precision(10);
        output << std::scientific;

        output << "# time (years): " << time << "\n";
        output << "# pos_x pos_y";
        if (spacedim == 3)
            output << " pos_z";
        output << " velocity_x velocity_y";
        if (spacedim == 3)
            output << " velocity_z";
        output << " plate_displacement crust_half_thickness sill_thickness";
        // if (unit_testing)
        //   output << " thickness_error";
        output << "\n";

        const QGauss<dim> quadrature_formula (flexure_degree+1);
        FEValues<dim,spacedim> fe_values (mapping, fe, quadrature_formula,
                                          update_values | update_quadrature_points | update_JxW_values);
        const unsigned int n_q_points = quadrature_formula.size ();

        std::vector<double> w_values(n_q_points);
        std::vector<double> h_values(n_q_points);
        std::vector<double> s_values(n_q_points);
        std::vector<Tensor<1,spacedim>> u_values(n_q_points);

        // compute the integral quantities by quadrature
        for (const auto &cell : dof_handler.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              fe_values.reinit (cell);
              fe_values[h_extractor].get_function_values (locally_relevant_solution, h_values);
              fe_values[w_extractor].get_function_values (locally_relevant_solution, w_values);
              fe_values[u_extractor].get_function_values (locally_relevant_solution, u_values);
              fe_values[s_extractor].get_function_values (locally_relevant_solution, s_values);

              for (unsigned int q=0; q<n_q_points; ++q) {
                const Point<spacedim> loc = fe_values.quadrature_point(q);
                output << loc[0] << " " << loc[1];
                if (spacedim == 3)
                    output << " " << loc[2];
                output << " " << u_values[q][0] << " " << u_values[q][1];
                if (spacedim == 3)
                    output << " " << u_values[q][2];
                output << " " << w_values[q] << " " << h_values[q]
                       << " " << s_values[q];
                output << "\n";
              }
            }
      }
    }
  }

  template <int spacedim>
  std::vector<double>
  CrustalFlow<spacedim>::
  calculate_error ()
  {
    double local_u_error_integral = 0.0;
    double local_w_error_integral = 0.0;
    double local_h_error_integral = 0.0;
    const QGauss<dim> quadrature_formula (flexure_degree+1);
    FEValues<dim,spacedim> fe_values (mapping, fe, quadrature_formula,
                                      update_values | update_quadrature_points | update_JxW_values);
    const unsigned int n_q_points = quadrature_formula.size ();

    std::vector<double> w_values(n_q_points);
    std::vector<double> h_values(n_q_points);
    std::vector<Tensor<1,spacedim>> u_values(n_q_points);

    // compute the integral quantities by quadrature
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit (cell);
          fe_values[w_extractor].get_function_values (locally_relevant_solution, w_values);
          fe_values[u_extractor].get_function_values (locally_relevant_solution, u_values);
          fe_values[h_extractor].get_function_values (locally_relevant_solution, h_values);

          for (unsigned int q=0; q<n_q_points; ++q) {
            const Point<spacedim> loc = fe_values.quadrature_point (q);
            const double X = loc[0];

            const double omega = test_perturbation_freq*dealii::numbers::PI/model_width;
            const double exact_w = 0.1*std::sin(omega*X) + 1.0;
            const double exact_u = 0.015*(omega)/(omega*omega + 2)*std::cos(omega*X);
            const double h_0 = initial_crustal_thickness_field.value(loc);
            const double exact_h = h_0 + crustal_flow_time * 2.0/3.0*h_0
                                          * 0.015*(omega*omega)/(omega*omega + 2)
                                          * std::sin(omega*X);

            const double err_w = std::pow(w_values[q] - exact_w, 2);
            const double err_u = std::pow(u_values[q][0] - exact_u, 2);
            const double err_h = std::pow(h_values[q] - exact_h, 2);
            local_w_error_integral += err_w*fe_values.JxW(q);
            local_u_error_integral += err_u*fe_values.JxW(q);
            local_h_error_integral += err_h*fe_values.JxW(q);
          }
        }
    // compute the sum over all processors
    double global_w_error_integral = std::sqrt(Utilities::MPI::sum (local_w_error_integral, MPI_COMM));
    double global_u_error_integral = std::sqrt(Utilities::MPI::sum (local_u_error_integral, MPI_COMM));
    double global_h_error_integral = std::sqrt(Utilities::MPI::sum (local_h_error_integral, MPI_COMM));

    std::vector<double> ret {global_w_error_integral, global_u_error_integral, global_h_error_integral};
    return ret;
    //return std::make_pair (global_w_error_integral, global_u_error_integral);
  }

template <int spacedim>
void
CrustalFlow<spacedim>::
print_step_header (const double timestep,
                   const double time)
  {
    std::locale s = pcout.get_stream().getloc();
    try { pcout.get_stream().imbue(std::locale(std::locale(), new ThousandSep)); }
    catch (const std::runtime_error &e) {}

    pcout << "\nTime step " << timestep << ": " << time << std::endl
          << "  Number of active cells: " << surface_mesh.n_global_active_cells()
          << " (on " << surface_mesh.n_levels() << " levels)" << std::endl
          << "  Number of degrees of freedom: " << dof_handler.n_dofs()
          << std::endl;

    pcout.get_stream().imbue(s);
  }

template <int spacedim>
void
CrustalFlow<spacedim>::
run ()
  {
    crustal_flow_timestep = 0;
    crustal_flow_time = 0;
    initialization_step = true;
    vis_timestep = 0;

    this-> initial_crustal_thickness_field.set_model_width(model_width);
    this-> topographic_boundary_value_field.set_model_width(model_width);
    this-> prescribed_overburden_field.set_model_width(model_width);

    GridGenerator::hyper_cube(surface_mesh, 0, model_width, false);
    surface_mesh.refine_global(initial_refinement);
    setup_dofs ();
    //apply_initial_conditions ();

    if (Utilities::MPI::this_mpi_process(MPI_COMM) == 0)
      pcout << "\nInitializing model" << std::endl;

    for (unsigned int i=0; i<(max_refinement-min_refinement); ++i)
      {
        if (Utilities::MPI::this_mpi_process(MPI_COMM) == 0)
          {
            pcout << "\nInitial refinement step "
              << Utilities::int_to_string(i+1) << " of "
              << Utilities::int_to_string(max_refinement-min_refinement)
              << std::endl;
          }

        assemble_system(0);
        this->use_direct_solver ? solve_direct() : solve_iterative();
        refine_mesh ();
      }

      assemble_system(0);
      this->use_direct_solver ? solve_direct() : solve_iterative();
      this->initialization_step = false;

      output_results(vis_timestep++, 0);

      do
        {
          print_step_header(crustal_flow_timestep, crustal_flow_time);

          const double max_dt = std::min(maximum_time_step,
                                         (end_time-crustal_flow_time));
          const double dt = get_dt(max_dt);

          {
            viscosity_field.set_time(crustal_flow_time);
            rigidity_field.set_time(crustal_flow_time);
            sill_emplacement_field.set_time(crustal_flow_time);
            initial_sill_thickness_field.set_time(crustal_flow_time);
          }

          double residual_norm;
          std::vector<double> h_residuals, w_residuals;
          do
            {
              assemble_system(dt);
              const std::pair<double, double> res = use_direct_solver ?
                                                    solve_direct() :
                                                    solve_iterative();
              h_residuals.push_back(res.first);
              w_residuals.push_back(res.second);
              residual_norm = (res.first+res.second)/(std::pow(model_width,spacedim-1));
            }
          while (residual_norm > picard_tolerance
                 && h_residuals.size() < max_nonlinear_iterations);

          pcout << "Picard iterations: " << h_residuals.size() << std::endl;
          {
            pcout << "h residuals: ";
            for (unsigned int i=0; i<h_residuals.size(); ++i)
              {
                pcout << h_residuals[i];
                if (i < h_residuals.size()-1)
                  pcout << " ";
              }
            pcout << std::endl;
          }
          {
            pcout << "w residuals: ";
            for (unsigned int i=0; i<w_residuals.size(); ++i)
              {
                pcout << w_residuals[i];
                if (i < w_residuals.size()-1)
                  pcout << " ";
              }
            pcout << std::endl;
          }

          if (unit_testing)
            {
              std::vector<double> errors = calculate_error();
              statistics.add_value("Flexure_Error", errors[0]);
              statistics.set_precision ("Flexure_Error", 8);
              statistics.set_scientific ("Flexure_Error", true);
              statistics.add_value("Velocity_Error", errors[1]);
              statistics.set_precision ("Velocity_Error", 8);
              statistics.set_scientific ("Velocity_Error", true);
              statistics.add_value("Thickness_Error", errors[2]);
              statistics.set_precision ("Thickness_Error", 8);
              statistics.set_scientific ("Thickness_Error", true);
            }

          crustal_flow_timestep += 1;
          crustal_flow_time += dt;

          if (crustal_flow_timestep % vis_frequency == 0)
            {
              output_results(vis_timestep++,
                             crustal_flow_time/YEAR_IN_SECONDS);
            }
        }
      while (crustal_flow_time < end_time);
  }
