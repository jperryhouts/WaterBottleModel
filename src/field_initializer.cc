#include "field_initializer.h"
#include <deal.II/base/utilities.h>

#include <cstdio>

template <int spacedim>
FieldInitializer<spacedim>::
FieldInitializer (const unsigned int n_components)
    : Functions::ParsedFunction<spacedim>(n_components)
  {}

template <int spacedim>
void
FieldInitializer<spacedim>::
declare_parameters(ParameterHandler &prm,
                   const std::string &default_val_as_string)
  {
    prm.declare_entry ("Source", "constant", Patterns::Selection("function|constant|ascii"),
                       "The FieldInitializer can take values from several sources: either "
                       "function, constant, or ascii.\n\nIn `function` mode, it will act as "
                       "an ordinary ParsedFunction, taking values in the `Function` "
                       "subsection below.\n\nIn `constant` mode, it will initialize itself "
                       "with a constant function (defined in the `Constant` parameter).\n\n"
                       "And in the `ascii` mode, it will read data from a text file, with "
                       "configuration defined in the `Ascii` subsection below.");

    prm.declare_entry ("Constant", default_val_as_string, Patterns::Double(0),
                       "If `Source = constant` is defined here, then the function will "
                       "always return this value, independent of location or time.");

    prm.enter_subsection ("Ascii");
    {
      prm.declare_entry ("Data file", "", Patterns::FileName(),
                         "Ascii text file containing gridded input data. Data along the "
                         "X-axis is delimited with newlines, while the Y-axis contains "
                         "individual numeric values, separated by other whitespace.\n\n"
                         "1D meshes (for models representing 2 spatial dimensions) would "
                         "be represented as a single column of numbers, separated by "
                         "line breaks.\n\n"
                         "The first and last data points in each dimension lie at the "
                         "domain edges, and all other nodes are distributed evenly across "
                         "the interior of the domain.");

      prm.declare_entry ("Bounds", "0, 0; 0, 0",
                         Patterns::List(Patterns::List(Patterns::Double(),
                                                       spacedim-1, spacedim, ","),
                                        2, 2, ";"),
                          "Minimum and maximum spatial bounds of the points defined "
                          "in this data file. Format is: \"min_x, min_y ; max_x, max_y\" "
                          "if in 3D spatial dimention, and \"min_x ; max_x\" otherwise.\n"
                          "Data within the ascii grid must be evenly spaced in Cartesian "
                          "coordinates between these bounds.\n\n"
                          "If all four values are set to zero, then it is assumed that "
                          "the input data spans the model.");
    }
    prm.leave_subsection ();

    prm.enter_subsection ("Function");
    {
      // Must be scalar-valued, even though n_components may be higher than 1
      Functions::ParsedFunction<spacedim>::declare_parameters (prm, 1);
    }
    prm.leave_subsection ();
  }

template <int spacedim>
void
FieldInitializer<spacedim>::
parse_parameters(ParameterHandler &prm)
  {
    const std::string src = prm.get("Source");

    if (src == "constant")
      {
        const double value = prm.get_double("Constant");
        this->initialize_constant_source(value);
      }
    else if (src == "ascii")
      {
        prm.enter_subsection ("Ascii");
        {
          const std::vector<std::string> bounds = Utilities::split_string_list(prm.get("Bounds"), ';');
          const std::vector<std::string> lwr = Utilities::split_string_list(bounds[0]);
          const std::vector<std::string> upr = Utilities::split_string_list(bounds[1]);

          Point<spacedim> lwr_bound, upr_bound;

          for (unsigned int i=0; i<lwr.size(); ++i)
            {
              lwr_bound[i] = Utilities::string_to_double(lwr[i]);
              upr_bound[i] = Utilities::string_to_double(upr[i]);
            }

          this->initialize_ascii_source(prm.get("Data file"),
                                        lwr_bound, upr_bound);
        }
        prm.leave_subsection();
      }
    else if (src == "function")
      {
        prm.enter_subsection ("Function");
        {
          Functions::ParsedFunction<spacedim>::parse_parameters (prm);
        }
        prm.leave_subsection();
      }
    else
      {
        AssertThrow(false, dealii::ExcMessage(std::string("Unrecognized source <")+ src + ">."));
      }
  }

template <int spacedim>
void
FieldInitializer<spacedim>::
set_model_width(const double width)
  {
    if (!(min_bound[0] || max_bound[0] || min_bound[1] || max_bound[1]))
      {
        // The bounds are not set, so we'll use the
        // model width as the upper bound.
        this->max_bound[0] = width;
        this->max_bound[1] = width;
      }
  }

template <int spacedim>
void
FieldInitializer<spacedim>::
set_time(const double)
  { /* Do nothing */ }

template <int spacedim>
void
FieldInitializer<spacedim>::
vector_value(const Point<spacedim> &p,
             Vector<double> &values) const
  {
    for (unsigned int comp=0; comp < this->n_components; ++comp)
      values(comp) = this->value(p, comp);
  }

template <int spacedim>
double
FieldInitializer<spacedim>::
value(const dealii::Point<spacedim> &p,
      unsigned int comp) const
  {
    switch (this->source)
      {
        case constant:
          return this->constant_value;
          ;;
        case function:
          return dealii::Functions::ParsedFunction<spacedim>::value (p, comp);
          ;;
        case ascii:
          unsigned int i=0, j=0;

          double scaled_x = (p[0] - this->min_bound[0]) /
                            (this->max_bound[0] - this->min_bound[0]);
          const unsigned int x_size = (unsigned int) this->data_grid.size();
          i = (unsigned int) round((x_size-1) * scaled_x);
          i = (i < x_size) ? i : (x_size-1);
          i = (i > 0) ? i : 0;

          // if (spacedim == 3) {
            double scaled_y = (p[1] - this->min_bound[1]) /
                              (this->max_bound[1] - this->min_bound[1]);
            const unsigned int y_size = (unsigned int) this->data_grid[0].size();
            j = (unsigned int) round((y_size-1) * scaled_y);
            j = (j < y_size) ? j : (y_size-1);
            j = (j > 0) ? j : 0;
          // }

          // std::cout << "point (" << p[0] << ", " << p[1] << ") = (" << i << ", " << j << ")." << std::endl;

          return this->data_grid[i][j];
          ;;
      }

      AssertThrow(false, ExcMessage("Unrecognized source type."));
      return 0; // Never reached
  }

template<int spacedim>
void
FieldInitializer<spacedim>::
initialize_constant_source (const double value)
  {
    this->source = constant;
    this->constant_value = value;
  }

template<int spacedim>
void
FieldInitializer<spacedim>::
initialize_ascii_source (const std::string &data_path,
                         Point<spacedim> min_bound,
                         Point<spacedim> max_bound)
  {
    this->source = ascii;

    this->min_bound = min_bound;
    this->max_bound = max_bound;

    AsciiUtils::load_data(data_path, this->data_grid);
  }

template class FieldInitializer<2>;
template class FieldInitializer<3>;