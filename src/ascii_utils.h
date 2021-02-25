
#ifndef ascii_utils_h
#define ascii_utils_h

#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.h>

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <iterator>
#include <cmath>

namespace AsciiUtils
{
  using namespace dealii;

  std::pair<unsigned int, unsigned int>
  count_rows_cols(const std::string &fname);

  std::string
  read_and_distribute_file_content(const std::string &filename);

  void
  load_data(const std::string &filename,
            std::vector<std::vector<double>> &matrix);

  class QuietException {};
}

#endif