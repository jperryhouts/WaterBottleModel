#include "ascii_utils.h"

namespace AsciiUtils
{
  using namespace dealii;

  std::pair<unsigned int, unsigned int>
  count_rows_cols(const std::string &filename)
    {
      // Start n_rows at 1 because we do an extra getline
      // call before we start incrementing it below.
      unsigned int n_rows = 1, n_cols = 0;

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          std::ifstream filestream(filename.c_str());
          if (!filestream)
          {
            unsigned int signaler = numbers::invalid_unsigned_int;
            const int ierr = MPI_Bcast(&signaler, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
            AssertThrowMPI(ierr);
            AssertThrow (false, ExcMessage(std::string("Could not open file <") + filename + ">."));
            return std::make_pair<unsigned int, unsigned int>(0,0); // Never happens
          }

          std::string line;

          std::getline(filestream, line);
          std::istringstream buf(line);
          std::istream_iterator<std::string> beg(buf), end;
          std::vector<std::string> tokens(beg, end);
          n_cols = tokens.size();

          while (std::getline(filestream, line))
            n_rows++;

          int ierr;
          ierr = MPI_Bcast(&n_rows,1,MPI_UNSIGNED,0,MPI_COMM_WORLD);
          AssertThrowMPI(ierr);
          ierr = MPI_Bcast(&n_cols,1,MPI_UNSIGNED,0,MPI_COMM_WORLD);
          AssertThrowMPI(ierr);
        }
      else
        {
          int ierr;
          ierr = MPI_Bcast(&n_rows,1,MPI_UNSIGNED,0,MPI_COMM_WORLD);
          AssertThrowMPI(ierr);
          ierr = MPI_Bcast(&n_cols,1,MPI_UNSIGNED,0,MPI_COMM_WORLD);
          AssertThrowMPI(ierr);
          if (n_rows == numbers::invalid_unsigned_int)
            throw QuietException();
        }

      return std::make_pair(n_rows, n_cols);
    }

  std::string
  read_and_distribute_file_content(const std::string &filename)
    {
      std::string data_string;

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
            // set file size to an invalid size (signaling an error if we can not read it)
            unsigned int filesize = numbers::invalid_unsigned_int;

            std::ifstream filestream(filename.c_str());

            if (!filestream)
            {
                const int ierr = MPI_Bcast(&filesize, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
                AssertThrowMPI(ierr);
                AssertThrow (false,
                             ExcMessage(std::string("Could not open file <") + filename + ">."));
                return data_string; // Never happens
            }

            std::stringstream datastream;
            filestream >> datastream.rdbuf();

            if (!filestream.eof())
            {
                MPI_Bcast(&filesize,1,MPI_UNSIGNED,0,MPI_COMM_WORLD);
                AssertThrow (false,
                            ExcMessage (std::string("Reading of file ") + filename + " finished " +
                                        "before the end of file was reached. Is the file corrupted or"
                                        "too large for the input buffer?"));
                return data_string; // Never happens
            }

            data_string = datastream.str();
            filesize = data_string.size();

            // Distribute data_size and data across processes
            int ierr = MPI_Bcast(&filesize,1,MPI_UNSIGNED,0,MPI_COMM_WORLD);
            AssertThrowMPI(ierr);
            ierr = MPI_Bcast(&data_string[0],filesize,MPI_CHAR,0,MPI_COMM_WORLD);
            AssertThrowMPI(ierr);
        }
      else
        {
            // Prepare for receiving data
            unsigned int filesize;
            int ierr = MPI_Bcast(&filesize,1,MPI_UNSIGNED,0,MPI_COMM_WORLD);
            AssertThrowMPI(ierr);
            if (filesize == numbers::invalid_unsigned_int)
              throw QuietException();

            data_string.resize(filesize);

            // Receive and store data
            ierr = MPI_Bcast(&data_string[0],filesize,MPI_CHAR,0,MPI_COMM_WORLD);
            AssertThrowMPI(ierr);
        }

      return data_string;
    }

    void
    load_data (const std::string &filename,
               std::vector<std::vector<double>> &matrix)
      {
        std::pair<unsigned int, unsigned int> data_dim
          = count_rows_cols (filename);
        const unsigned int n_rows = data_dim.first,
                           n_cols = data_dim.second;

        matrix.resize(n_rows, std::vector<double>(n_cols));

        std::stringstream in(read_and_distribute_file_content(filename));

        // std::cout << "Found " << n_rows << " rows, and " << n_cols << " cols." << std::endl;

        double tmp_value;
        unsigned int i, j;
        for (unsigned int idx=0; in >> tmp_value; ++idx)
          {
            i = idx/n_rows;
            j = idx%n_cols;
            matrix[i][j] = tmp_value;
          }
      }
}