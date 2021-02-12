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

#include "parameters.h"

std::string
read_until_end (std::istream &input)
{
  std::string result;
  while (input)
    {
      std::string line;
      std::getline(input, line);

      result += line + '\n';
    }
  return result;
}

// get the value of a particular parameter from the contents of the input
// file. return an empty string if not found
std::string
get_last_value_of_parameter(const std::string &parameters,
                            const std::string &parameter_name)
{
  std::string return_value;

  std::istringstream x_file(parameters);
  while (x_file)
    {
      // get one line and strip spaces at the front and back
      std::string line;
      std::getline(x_file, line);
      while ((line.size() > 0) && (line[0] == ' ' || line[0] == '\t'))
        line.erase(0, 1);
      while ((line.size() > 0)
             && (line[line.size() - 1] == ' ' || line[line.size() - 1] == '\t'))
        line.erase(line.size() - 1, std::string::npos);
      // now see whether the line starts with 'set' followed by multiple spaces
      // if not, try next line
      if (line.size() < 4)
        continue;

      if ((line[0] != 's') || (line[1] != 'e') || (line[2] != 't')
          || !(line[3] == ' ' || line[3] == '\t'))
        continue;

      // delete the "set " and then delete more spaces if present
      line.erase(0, 4);
      while ((line.size() > 0) && (line[0] == ' ' || line[0] == '\t'))
        line.erase(0, 1);
      // now see whether the next word is the word we look for
      if (line.find(parameter_name) != 0)
        continue;

      line.erase(0, parameter_name.size());
      while ((line.size() > 0) && (line[0] == ' ' || line[0] == '\t'))
        line.erase(0, 1);

      // we'd expect an equals size here
      if ((line.size() < 1) || (line[0] != '='))
        continue;

      // remove comment
      std::string::size_type pos = line.find('#');
      if (pos != std::string::npos)
        line.erase (pos);

      // trim the equals sign at the beginning and possibly following spaces
      // as well as spaces at the end
      line.erase(0, 1);
      while ((line.size() > 0) && (line[0] == ' ' || line[0] == '\t'))
        line.erase(0, 1);
      while ((line.size() > 0) && (line[line.size()-1] == ' ' || line[line.size()-1] == '\t'))
        line.erase(line.size()-1, std::string::npos);

      // the rest should now be what we were looking for
      return_value = line;
    }

  return return_value;
}

/**
 * Let ParameterHandler parse the input file, here given as a string.
 * Since ParameterHandler unconditionally writes to the screen when it
 * finds something it doesn't like, we get massive amounts of output
 * in parallel computations since every processor writes the same
 * stuff to screen. To avoid this, let processor 0 parse the input
 * first and, if necessary, produce its output. Only if this
 * succeeds, also let the other processors read their input.
 *
 * In case of an error, we need to abort all processors without them
 * having read their data. This is done by throwing an exception of the
 * special class QuietException that we can catch in main() and terminate
 * the program quietly without generating other output.
 */
void
read_parameter_file (const std::string &input_as_string,
                     dealii::ParameterHandler  &prm)
{
  // try reading on processor 0
  bool success = true;
  if (dealii::Utilities::MPI::this_mpi_process (MPI_COMM_WORLD) == 0)
    try
      {
        prm.parse_input_from_string(input_as_string.c_str());
      }
    catch (const dealii::ExceptionBase &e)
      {
        success = false;
        e.print_info(std::cerr);
        std::cerr << std::endl;
      }


  // broadcast the result. we'd like to do this with a bool
  // data type but MPI_C_BOOL is not part of old MPI standards.
  // so, do the broadcast in integers
  {
    int isuccess = (success ? 1 : 0);
    const int ierr = MPI_Bcast (&isuccess, 1, MPI_INT, 0, MPI_COMM_WORLD);
    AssertThrowMPI(ierr);
    success = (isuccess == 1);
  }

  // if not success, then throw an exception: ExcMessage on processor 0,
  // QuietException on the others
  if (success == false)
    {
      if (dealii::Utilities::MPI::this_mpi_process (MPI_COMM_WORLD) == 0)
        {
          AssertThrow(false, dealii::ExcMessage ("Invalid input parameter file."));
        }
      else
        throw QuietException();
    }

  // otherwise, processor 0 was ok reading the data, so we can expect the
  // other processors will be ok as well
  if (dealii::Utilities::MPI::this_mpi_process (MPI_COMM_WORLD) != 0)
    {
      prm.parse_input_from_string(input_as_string.c_str());
    }
}

template <int spacedim>
void
declare_parameters (dealii::ParameterHandler &prm)
  {
    prm.declare_entry ("Unit testing", "false", Patterns::Bool(), "");
    prm.declare_entry ("Test perturbation frequency", "2",   Patterns::Integer(1), "");

    prm.declare_entry ("Dimension", "2", Patterns::Integer(0), "");
    prm.declare_entry ("Output directory", "output", Patterns::DirectoryName(), "");
    prm.declare_entry ("End time", "1e10", Patterns::Double(0), "");
    prm.declare_entry ("Maximum time step", "1e10", Patterns::Double(0), "");

    prm.declare_entry ("Model width", "1.0", Patterns::Double(0), "");
    prm.declare_entry ("Output format", "vtu", Patterns::Selection("vtu|ascii|none"), "");

    prm.declare_entry ("CFL", "0.1", Patterns::Double(0), "");
    prm.declare_entry ("Use direct solver", "false", Patterns::Bool(), "");
    prm.declare_entry ("Linear solver tolerance", "1e-12", Patterns::Double(0), "");
    prm.declare_entry ("Picard tolerance", "1e-12", Patterns::Double(0), "");
    prm.declare_entry ("Initial refinement", "4", Patterns::Integer(0), "");
    prm.declare_entry ("Minimum refinement", "3", Patterns::Integer(0), "");
    prm.declare_entry ("Maximum refinement", "6", Patterns::Integer(0), "");
    prm.declare_entry ("Visualization frequency", "10", Patterns::Integer(0), "");
    prm.declare_entry ("Initial crustal thickness", "1e3", Patterns::Double(0), "");
    prm.declare_entry ("Max nonlinear iterations", "50", Patterns::Integer(0), "");

    prm.declare_entry ("Gravity", "9.8", Patterns::Double(0), "");

    prm.declare_entry ("Pi", "1.0", Patterns::Double(0),
                       "Dimensionless parameter, $\\Pi$.");

    prm.declare_entry ("gamma", "1.0", Patterns::Double(0),
                       "Dimensionless parameter, $\\gamma$.");

    prm.declare_entry ("Density of mantle", "1.0", Patterns::Double(0),
                       "Density of the mantle.");

    prm.declare_entry ("Density of crust", "0.80", Patterns::Double(0),
                       "Density of the crust, including the lower crust "
                       "and overlying elastic upper crust.");

    prm.declare_entry ("Density of overburden", "0.27", Patterns::Double(0),
                       "Density of overburden load. In the case of a mid-"
                       "crustal sill, this is assumed to be the density "
                       "in excess of ordinary crust.");

    prm.declare_entry ("Coefficient functions coordinate system", "cartesian",
                       Patterns::Selection ("cartesian|spherical|depth"),
                       "A selection that determines the assumed coordinate "
                       "system for the function variables. Allowed values "
                       "are `cartesian', `spherical', and `depth'. `spherical' coordinates "
                       "are interpreted as r,phi or r,phi,theta in 2D/3D "
                       "respectively with theta being the polar angle. `depth' "
                       "will create a function, in which only the first "
                       "parameter is non-zero, which is interpreted to "
                       "be the depth of the point.");

    prm.enter_subsection("Rigidity function");
    {
      Functions::ParsedFunction<spacedim>::declare_parameters (prm, 1);
    }
    prm.leave_subsection();

    prm.enter_subsection("Crustal viscosity function");
    {
      Functions::ParsedFunction<spacedim>::declare_parameters (prm, 1);
    }
    prm.leave_subsection();

    prm.enter_subsection("Sill emplacement function");
    {
      Functions::ParsedFunction<spacedim>::declare_parameters (prm, 1);
    }
    prm.leave_subsection();

    prm.enter_subsection("Sill thickness function");
    {
      Functions::ParsedFunction<spacedim>::declare_parameters (prm, 1);
    }
    prm.leave_subsection();
  }

template <int spacedim>
void
parse_parameters (dealii::ParameterHandler &prm,
                  CrustalFlow<spacedim> &model)
  {
    model.end_time = prm.get_double("End time");
    model.maximum_time_step = prm.get_double("Maximum time step");

    model.Pi = prm.get_double("Pi");

    model.gamma = prm.get_double("gamma");

    model.rho_m = prm.get_double("Density of mantle");

    model.rho_c = prm.get_double("Density of crust");

    model.rho_s = prm.get_double("Density of overburden");

    model.gravity = prm.get_double("Gravity");

    model.h_0 = prm.get_double("Initial crustal thickness");
    model.w_0 = 2 * (model.rho_m-model.rho_c) * model.h_0 / model.rho_m; /* +(rho_m-rho_s)*S_0 */

    model.solver_relative_tolerance = prm.get_double("Linear solver tolerance");
    model.picard_tolerance = prm.get_double("Picard tolerance");

    model.output_format = prm.get("Output format");
    model.model_width = prm.get_double("Model width");

    model.unit_testing = prm.get_bool("Unit testing");
    if (model.unit_testing)
    {
      model.h_0 = 1.0;
      model.w_0 = 1.0;
      model.gravity = 1.0;
      model.test_perturbation_freq = prm.get_integer("Test perturbation frequency");
    }

    model.CFL = prm.get_double("CFL");
    model.use_direct_solver = prm.get_bool("Use direct solver");
    model.initial_refinement = prm.get_integer("Initial refinement");
    model.min_refinement = prm.get_integer("Minimum refinement");
    model.max_refinement = prm.get_integer("Maximum refinement");
    AssertThrow(model.max_refinement >= model.min_refinement,
                ExcMessage("Maximum refinement level cannot be lower than "
                            "minimum refinement level"));

    model.vis_frequency = prm.get_integer("Visualization frequency");

    model.max_nonlinear_iterations = prm.get_integer("Max nonlinear iterations");

    // velocity_degree = prm.get_double("Velocity polynomial degree");
    // flexure_degree = prm.get_double("Flexure polynomial degree");

    // model.coefficient_function_coordinate_system =
    //   Utilities::Coordinates::string_to_coordinate_system(
    //     prm.get("Coefficient functions coordinate system"));

    prm.enter_subsection("Rigidity function");
    {
      try
        {
          model.rigidity_function.parse_parameters (prm);
        }
      catch (...)
        {
          std::cerr << "ERROR: FunctionParser failed to parse\n"
                    << "\t'Postprocess.Crustal flow.Rigidity function'\n"
                    << "with expression\n"
                    << "\t'" << prm.get("Function expression") << "'\n"
                    << "More information about the cause of the parse error \n"
                    << "is shown below.\n";
          throw;
        }
    }
    prm.leave_subsection();

    prm.enter_subsection("Crustal viscosity function");
    {
      try
        {
          model.viscosity_function.parse_parameters (prm);
        }
      catch (...)
        {
          std::cerr << "ERROR: FunctionParser failed to parse\n"
                    << "\t'Postprocess.Crustal flow.Crustal viscosity function'\n"
                    << "with expression\n"
                    << "\t'" << prm.get("Function expression") << "'"
                    << "More information about the cause of the parse error \n"
                    << "is shown below.\n";
          throw;
        }
    }
    prm.leave_subsection();

    prm.enter_subsection("Sill emplacement function");
    {
      try
        {
          model.sill_emplacement_function.parse_parameters (prm);
        }
      catch (...)
        {
          std::cerr << "ERROR: FunctionParser failed to parse\n"
                    << "\t'Postprocess.Crustal flow.Sill emplacement function'\n"
                    << "with expression\n"
                    << "\t'" << prm.get("Function expression") << "'"
                    << "More information about the cause of the parse error \n"
                    << "is shown below.\n";
          throw;
        }
    }
    prm.leave_subsection();

    prm.enter_subsection("Sill thickness function");
    {
      try
        {
          model.sill_thickness_function.parse_parameters (prm);
        }
      catch (...)
        {
          std::cerr << "ERROR: FunctionParser failed to parse\n"
                    << "\t'Postprocess.Crustal flow.Sill emplacement function'\n"
                    << "with expression\n"
                    << "\t'" << prm.get("Function expression") << "'"
                    << "More information about the cause of the parse error \n"
                    << "is shown below.\n";
          throw;
        }
    }
    prm.leave_subsection();
  }

  template void declare_parameters<2> (dealii::ParameterHandler &prm);
  template void parse_parameters<2> (dealii::ParameterHandler &prm,
                                     CrustalFlow<2> &model);

  template void declare_parameters<3> (dealii::ParameterHandler &prm);
  template void parse_parameters<3> (dealii::ParameterHandler &prm,
                                     CrustalFlow<3> &model);
