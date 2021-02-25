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

#include "main.h"

template <int spacedim>
void
run_model(const std::string &input_as_string)
{
  CrustalFlow<spacedim> model(MPI_COMM_WORLD);

  dealii::ParameterHandler prm;
  declare_parameters<spacedim>(prm);
  read_parameter_file(input_as_string, prm);
  parse_parameters<spacedim>(prm, model);

  { // Initialize model object variables
    model.set_outputdir(prm.get("Output directory"));

    model.statistics.set_auto_fill_mode(true);

    int error;
    if ((Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
      {
        std::ofstream duplicate(model.output_directory + "original.prm");
        duplicate << input_as_string;

        const std::string solution_directory
          = model.output_directory + "crustal_flow";
        DIR *dir = opendir(solution_directory.c_str());
        if (dir == nullptr)
          error = mkdirp (solution_directory,
                          S_IRWXU|S_IRGRP|S_IXGRP|S_IROTH|S_IXOTH);
        else
          error = closedir(dir);
        MPI_Bcast (&error, 1, MPI_INT, 0, MPI_COMM_WORLD);
        AssertThrow(error == 0,
                    ExcMessage(std::string("Can't create the output directory at <") + solution_directory + ">"));

        model.log_file_stream.open((model.output_directory + "log.txt").c_str(),
                                   std::ios_base::out);

        print_run_header(model.log_file_stream);
      }
    else
      {
        MPI_Bcast (&error, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (error!=0)
          throw QuietException();
      }
  }

  model.run();
}

int
main (int argc, char *argv[])
{
  bool do_print_license = false;
  bool do_print_help = false;

  try
    {
      int n_mpi_args = 0;
      char **mpi_args = nullptr;
      Utilities::MPI::MPI_InitFinalize mpi_initialization (
          n_mpi_args, mpi_args, 1);

      if ((Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
        print_run_header(std::cout);

      if (do_print_license)
        {
          print_license();
          return 0;
        }
      else if (do_print_help)
        {
          print_help();
          return 0;
        }

      // Read the input parameter file to a string.
      MPIAssertThrow(argc >= 2, "No input file specified");
      const std::string parameter_file_name = argv[argc-1];

      struct stat buffer;
      MPIAssertThrow(stat (parameter_file_name.c_str(), &buffer) == 0,
                     "Parameter file does not exist.");

      const std::string input_as_string =
        AsciiUtils::read_and_distribute_file_content(parameter_file_name);

      const std::string dimstr =
        get_last_value_of_parameter(input_as_string, "Dimension");
      const unsigned int dim = dealii::Utilities::string_to_int (dimstr);

      switch (dim)
        {
          case 2:
          {
            run_model<2>(input_as_string);
            break;
          }
          case 3:
          {
            run_model<3>(input_as_string);
            break;
          }
          default:
            AssertThrow((dim >= 2) && (dim <= 3),
                        ExcMessage ("Incompatible dimension given"));
        }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl << "Exception on processing: " << std::endl
                << exc.what () << std::endl << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (QuietException &)
    {
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl << "Unknown exception!" << std::endl << "Aborting!"
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
