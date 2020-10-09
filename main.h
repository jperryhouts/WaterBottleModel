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

inline
std::string
get_timestamp ()
{
  std::time_t rawtime;
  struct tm * timeinfo;
  char buffer [80];
  std::time (&rawtime);
  timeinfo = std::localtime (&rawtime);
  strftime (buffer,80,"%Y%m%d%H%M%S",timeinfo);
  const std::string timestamp (buffer);
  return timestamp;
}

inline
bool
file_exists(const std::string &fname)
{
  struct stat buffer;
  return (stat (fname.c_str(), &buffer) == 0);
}

inline
int
mkdirp(std::string pathname,const mode_t mode)
{
  // force trailing / so we can handle everything in loop
  if (pathname[pathname.size()-1] != '/')
    {
      pathname += '/';
    }

  size_t pre = 0;
  size_t pos;

  while ((pos = pathname.find_first_of('/',pre)) != std::string::npos)
    {
      const std::string subdir = pathname.substr(0,pos++);
      pre = pos;

      // if leading '/', first string is 0 length
      if (subdir.size() == 0)
        continue;

      int mkdir_return_value;
      if ((mkdir_return_value = mkdir(subdir.c_str(),mode)) && (errno != EEXIST))
        return mkdir_return_value;

    }

  return 0;
}

inline
void
print_help ()
  {
    std::cout << "\nUsage: ./wbm [args] <parameter_file.prm>\n\n"
              << "    optional arguments [args]:\n"
              << "        -h, --help            (show this message and exit)\n"
              << "        -l, --license         (show information about modifying\n"
              << "                               and redistributing this code)\n"
              << std::endl;
  }

inline
void
print_license ()
  {
    std::cout << "\nCopyright (C) 2020 Jonathan Perry-Houts\n\n"
              << "This program is free software: you can redistribute it and/or modify\n"
              << "it under the terms of the GNU General Public License as published by\n"
              << "the Free Software Foundation, either version 3 of the License, or\n"
              << "(at your option) any later version.\n"
              << "\n"
              << "This program is distributed in the hope that it will be useful,\n"
              << "but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
              << "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n"
              << "GNU General Public License for more details.\n"
              << "\n"
              << "You should have received a copy of the GNU General Public License\n"
              << "along with this program.  If not, see <https://www.gnu.org/licenses/>.\n"
              << std::endl;
  }

template <class Stream>
void
print_run_header (Stream &stream)
  {
    const int n_tasks = dealii::Utilities::MPI::n_mpi_processes (MPI_COMM_WORLD);
    stream << "-----------------------------------------------------------------------------"
           << std::endl;
    stream << "--  Water Bottle Model for lower crustal flow\n";
    stream << "--     . using deal.II " << DEAL_II_PACKAGE_VERSION;
    stream << "\n";
    stream << "--     .       with "
#ifdef DEAL_II_WITH_64BIT_INDICES
           << "64"
#else
           << "32"
#endif
           << " bit indices and vectorization level ";

#if DEAL_II_VERSION_GTE(9,2,0)
  const unsigned int n_vect_bits =
    dealii::VectorizedArray<double>::size() * 8 * sizeof(double);
#else
  const unsigned int n_vect_bits =
    dealii::VectorizedArray<double>::n_array_elements * 8 * sizeof(double);
#endif

    stream << DEAL_II_COMPILER_VECTORIZATION_LEVEL << " (" << n_vect_bits << " bits)\n";

    stream << "--     . using Trilinos " << DEAL_II_TRILINOS_VERSION_MAJOR
           << '.' << DEAL_II_TRILINOS_VERSION_MINOR << '.'
           << DEAL_II_TRILINOS_VERSION_SUBMINOR << '\n';

#ifdef DEBUG
    stream << "--     . running in DEBUG mode"
#else
    stream << "--     . running in OPTIMIZED mode"
#endif
           << " with "
           << n_tasks << " MPI process" << (n_tasks == 1 ? "\n" : "es\n");
    stream << "-----------------------------------------------------------------------------"
           << std::endl;
  }

template void print_run_header<std::ostream> (std::ostream &stream);
template void print_run_header<std::ofstream> (std::ofstream &stream);
