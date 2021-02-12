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

#ifndef _model_h
#define _model_h

#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/signaling_nan.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/shared_tria.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <csignal>

#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>

#include <boost/iostreams/tee.hpp>
#include <boost/iostreams/stream.hpp>

#include <deal.II/base/parsed_function.h>

constexpr double YEAR_IN_SECONDS = 60*60*24*365.2425;

namespace LA
{
  using namespace dealii::LinearAlgebraTrilinos;
}

using namespace dealii;

struct ThousandSep : std::numpunct<char>
{
  protected:
    virtual char do_thousands_sep() const
    {
      return ',';
    }
    virtual std::string do_grouping() const
    {
      return "\003";  // groups of 3 digits (this string is in octal format)
    }
};

template <int spacedim>
class CrustalFlow
{
  public:
    /**
     * Constructor.
     */
    CrustalFlow (const MPI_Comm mpi_communicator_);

    /*
     * Destructor
     */
    ~CrustalFlow ();

    void run ();

    void set_outputdir(const std::string &dirname);

  //private:
    static constexpr unsigned int dim = spacedim - 1;

    MPI_Comm MPI_COMM;

    std::ofstream log_file_stream;
    typedef boost::iostreams::tee_device<std::ostream, std::ofstream> TeeDevice;
    typedef boost::iostreams::stream< TeeDevice > TeeStream;
    TeeDevice iostream_tee_device;
    TeeStream iostream_tee_stream;

    ConditionalOStream pcout;
    TimerOutput computing_timer;

    std::string output_directory;

    TableHandler      statistics;
    Threads::Thread<> output_statistics_thread;
    std::size_t       statistics_last_write_size;
    std::size_t       statistics_last_hash;

    static constexpr unsigned int velocity_degree = 1;
    static constexpr unsigned int flexure_degree = 2;
    static constexpr unsigned int h_degree = 1;
    static constexpr unsigned int s_degree = 1;

    static constexpr unsigned int u_component_index = 0;
    static constexpr unsigned int h_component_index = spacedim;
    static constexpr unsigned int v_component_index = spacedim+1;
    static constexpr unsigned int w_component_index = spacedim+2;
    static constexpr unsigned int s_component_index = spacedim+3;

    parallel::distributed::Triangulation<dim, spacedim> surface_mesh;

    std::vector<typename DoFHandler<spacedim>::active_cell_iterator> local_boundary_cells;

    MappingQ<dim, spacedim>   mapping;
    DoFHandler<dim, spacedim> dof_handler;
    FESystem<dim, spacedim>   fe;
    AffineConstraints<double> constraints;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    LA::MPI::SparseMatrix system_matrix;
    LA::MPI::Vector locally_relevant_solution;
    LA::MPI::Vector old_locally_relevant_solution;
    LA::MPI::Vector rhs;

    const FEValuesExtractors::Vector u_extractor;
    const FEValuesExtractors::Scalar h_extractor;
    const FEValuesExtractors::Scalar v_extractor;
    const FEValuesExtractors::Scalar w_extractor;
    const FEValuesExtractors::Scalar s_extractor;

    unsigned int crustal_flow_timestep;
    double crustal_flow_time;

    // Flow equation methods
    void output_statistics();
    void setup_dofs ();
    void do_initial_timestep ();
    void assemble_system (const double dt);
    std::pair<double, double> solve_direct ();
    std::pair<double, double> solve_iterative ();
    std::pair<double, double> solve_bicgstab ();
    double picard_residual (const LA::MPI::Vector &distributed_solution,
                            const FEValuesExtractors::Scalar extractor,
                            const unsigned int degree);
    double get_dt (const double max_dt);
    void print_step_header(const double timestep,
                           const double time);

    std::vector<double> calculate_error();

    // Joint system methods: operate on shared mesh and/or both FE solutions
    void refine_mesh ();
    void output_results (const unsigned int timestep,  const double time);

    double maximum_time_step;
    double Pi, gamma;
    double rho_c, rho_m, rho_s, gravity;
    double E=7e11, Te=10e3, nu=0.25;
    double RIGIDITY=((E * std::pow(Te, 3)) / (12 * (1.0 - nu *nu)));
    unsigned int vis_timestep;
    double w_0, h_0; // Initial condition (and boundary condition)
    bool use_direct_solver;
    double solver_relative_tolerance;
    double picard_tolerance;
    std::string output_format;
    double CFL;
    double model_width;
    unsigned int initial_refinement, min_refinement, max_refinement;
    bool initialization_step;
    unsigned int vis_frequency;
    unsigned int max_nonlinear_iterations;
    double end_time;

    bool unit_testing;
    double test_perturbation_freq;

    /**
     * A function object representing the effective elastic thickness.
     */
    Functions::ParsedFunction<spacedim> rigidity_function;
    Functions::ParsedFunction<spacedim> viscosity_function;
    Functions::ParsedFunction<spacedim> sill_emplacement_function;
    Functions::ParsedFunction<spacedim> sill_thickness_function;
    Functions::ParsedFunction<spacedim> initial_crustal_thickness_function;
    /**
     * The coordinate representation to evaluate the function. Possible
     * choices are depth, cartesian and spherical.
     */
    //Utilities::Coordinates::CoordinateSystem coefficient_function_coordinate_system;
};

template class CrustalFlow<2>;
template class CrustalFlow<3>;

class QuietException {};

#endif
