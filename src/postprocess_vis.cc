#include "postprocess_vis.h"

namespace Postprocessor
{
  template <int spacedim>
  StaticFunctionPostprocessor<spacedim>::
  StaticFunctionPostprocessor(const std::string &name,
                              FieldInitializer<spacedim>* function_ptr)
    : DataPostprocessorScalar<spacedim> (name, update_quadrature_points)
  {
    this->field = function_ptr;
  }

  template <int spacedim>
  void
  StaticFunctionPostprocessor<spacedim>::
  evaluate_vector_field(const DataPostprocessorInputs::Vector<spacedim> &input_data,
                        std::vector<Vector<double>> &computed_quantities) const
    {
      const unsigned int n_quadrature_points = input_data.solution_values.size();
      Assert (computed_quantities.size() == n_quadrature_points,          ExcInternalError());
      Assert (computed_quantities[0].size() == 1,                         ExcInternalError());

      for (unsigned int q=0; q<n_quadrature_points; ++q)
        {
          computed_quantities[q](0) = this->field->value(input_data.evaluation_points[q]);
        }
    }
}