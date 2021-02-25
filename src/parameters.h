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
#include "field_initializer.h"

#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <csignal>
#include <string>

std::string
get_last_value_of_parameter(const std::string &parameters,
                            const std::string &parameter_name);

void
read_parameter_file (const std::string &input_as_string,
                     dealii::ParameterHandler  &prm);

template <int spacedim>
void
declare_parameters (dealii::ParameterHandler &prm);

template <int spacedim>
void
parse_parameters (dealii::ParameterHandler &prm,
                  CrustalFlow<spacedim> &model);
