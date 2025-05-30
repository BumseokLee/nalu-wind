// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/ConductionDiagonal.h"

#include "matrix_free/Coefficients.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/ValidSimdLength.h"
#include "matrix_free/ShuffledAccess.h"
#include "matrix_free/KokkosViewTypes.h"
#include "ArrayND.h"

#include <KokkosInterface.h>
#include <Kokkos_ScatterView.hpp>
#include <stk_simd/Simd.hpp>

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace impl {
namespace {

template <int p, int dir, typename MetricType, typename LHSType>
KOKKOS_FUNCTION void
diffusion_diagonal(
  int index,
  const typename Coeffs<p>::nodal_matrix_type& vandermonde,
  const typename Coeffs<p>::nodal_matrix_type& nodal_derivative,
  const typename Coeffs<p>::scs_matrix_type& flux_point_derivative,
  const typename Coeffs<p>::scs_matrix_type& flux_point_interpolant,
  const MetricType& metric,
  LHSType& lhs)
{
  for (int l = 0; l < p; ++l) {
    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        const ftype Ws = vandermonde(s, s);
        const ftype Wr = vandermonde(r, r);
        const ftype orth = Ws * Wr * metric(index, dir, l, s, r, 0);
        ftype non_orth = 0;
        for (int q = 0; q < p + 1; ++q) {
          non_orth += Ws * vandermonde(r, q) * nodal_derivative(q, r) *
                        metric(index, dir, l, s, q, 1) +
                      Wr * vandermonde(s, q) * nodal_derivative(q, s) *
                        metric(index, dir, l, q, r, 2);
        }
        shuffled_access<dir>(lhs, s, r, l + 0) +=
          orth * flux_point_derivative(l, l + 0) +
          flux_point_interpolant(l, l + 0) * non_orth;
        shuffled_access<dir>(lhs, s, r, l + 1) -=
          orth * flux_point_derivative(l, l + 1) +
          flux_point_interpolant(l, l + 1) * non_orth;
      }
    }
  }
}

} // namespace

template <int p>
void
conduction_diagonal_t<p>::invoke(
  double gamma,
  const_elem_offset_view<p> offsets,
  const_scalar_view<p> volumes,
  const_scs_vector_view<p> metric,
  tpetra_view_type yout)
{
  auto yout_scatter = Kokkos::Experimental::create_scatter_view(yout);
  Kokkos::parallel_for(
    "diagonal", DeviceRangePolicy(0, offsets.extent_int(0)),
    KOKKOS_LAMBDA(int index) {
      constexpr auto flux_point_interpolant = Coeffs<p>::Nt;
      constexpr auto flux_point_derivative = Coeffs<p>::Dt;
      constexpr auto nodal_derivative = Coeffs<p>::D;
      constexpr auto vandermonde = Coeffs<p>::W;
      constexpr auto Wl = Coeffs<p>::Wl;

      ArrayND<ftype[p + 1][p + 1][p + 1]> lhs;
      for (int k = 0; k < p + 1; ++k) {
        const auto gammaWk = gamma * Wl(k);
        for (int j = 0; j < p + 1; ++j) {
          const auto gammaWkWj = gammaWk * Wl(j);
          for (int i = 0; i < p + 1; ++i) {
            lhs(k, j, i) = gammaWkWj * Wl(i) * volumes(index, k, j, i);
          }
        }
      }

      diffusion_diagonal<p, 0>(
        index, vandermonde, nodal_derivative, flux_point_derivative,
        flux_point_interpolant, metric, lhs);
      diffusion_diagonal<p, 1>(
        index, vandermonde, nodal_derivative, flux_point_derivative,
        flux_point_interpolant, metric, lhs);
      diffusion_diagonal<p, 2>(
        index, vandermonde, nodal_derivative, flux_point_derivative,
        flux_point_interpolant, metric, lhs);

      auto accessor = yout_scatter.access();
      const int valid_simd_len = valid_offset<p>(index, offsets);
      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          for (int i = 0; i < p + 1; ++i) {
            for (int n = 0; n < valid_simd_len; ++n) {
              accessor(offsets(index, k, j, i, n), 0) +=
                stk::simd::get_data(lhs(k, j, i), n);
            }
          }
        }
      }
    });
  Kokkos::Experimental::contribute(yout, yout_scatter);
}
INSTANTIATE_POLYSTRUCT(conduction_diagonal_t);
} // namespace impl

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
