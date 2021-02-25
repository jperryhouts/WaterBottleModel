#ifndef _field_initializer_h
#define _field_initializer_h

#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parsed_function.h>

#include "ascii_utils.h"

using namespace dealii;

template <int spacedim>
class FieldInitializer : public Functions::ParsedFunction<spacedim>
{
  public:
    /**
     * Constructor.
     */
    FieldInitializer(const unsigned int n_components = 1);

    static void
    declare_parameters(ParameterHandler &prm,
                       const std::string &default_val_as_string);

    void
    parse_parameters(ParameterHandler &prm);

    virtual void
    set_model_width(const double width);

    virtual double
    value(const Point<spacedim> &p,
          unsigned int component = 0) const override;

    virtual void
    vector_value(const Point<spacedim> &p,
                 Vector<double> &values) const override;

    virtual void
    set_time(const double newtime) override;

    void
    initialize_constant_source(const double value);

    void
    initialize_ascii_source(const std::string &data_path,
                            Point<spacedim> min_bound,
                            Point<spacedim> max_bound);

  private:

    enum SourceFlags {
        function = 0,
        ascii    = 1,
        constant = 2
    } source = function;

    // For constant value source
    double constant_value;

    // For the ASCII reader source
    /**
     * The coordinate values in each direction.
     * First row is the location of each node point in the 'x'
     * direction, and second row is each node point in the 'y'
     * direction (if we're in 3D).
     */
    std::vector<std::vector<double>> coordinate_values;

    /**
     * Gridded data at evenly spaced intervals. Each row corresponds
     * with an X position indicated in the first row of the
     * 'coordinates.txt' file. Each column is at a 'Y' position indicated
     * in the second row of the 'coordinates' file (if 3D).
     */
    std::vector<std::vector<double>> data_grid;

    Point<spacedim> min_bound, max_bound;
};

#endif