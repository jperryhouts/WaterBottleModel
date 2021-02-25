#ifndef _aspect_postprocess_visualization_h
#define _aspect_postprocess_visualization_h

#include <deal.II/base/function.h>
#include <deal.II/numerics/data_postprocessor.h>

#include "field_initializer.h"

namespace Postprocessor
{
  using namespace dealii;

  /*
   * Just a wrapper for outputting a static function in
   * vtu visualizations.
   */
  template <int spacedim>
  class StaticFunctionPostprocessor : public DataPostprocessorScalar<spacedim>
  {
    public:
      StaticFunctionPostprocessor(const std::string &name,
                                  FieldInitializer<spacedim>* function_ptr);

      void
      evaluate_vector_field(const DataPostprocessorInputs::Vector<spacedim> &input_data,
                            std::vector<Vector<double>> &computed_quantities) const override;

    private:
      FieldInitializer<spacedim>* field;
  };

  template class StaticFunctionPostprocessor<2>;
  template class StaticFunctionPostprocessor<3>;

}

#endif