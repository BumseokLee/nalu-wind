// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/SANuTildaNodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "SimdInterface.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

SANuTildaNodeKernel::SANuTildaNodeKernel(const stk::mesh::MetaData& meta)
  : NGPNodeKernel<SANuTildaNodeKernel>(),
    nuTildaID_(get_field_ordinal(meta, "sa_nu_tilda")),
    densityID_(get_field_ordinal(meta, "density")),
    viscosityID_(get_field_ordinal(meta, "viscosity")),
    dudxID_(get_field_ordinal(meta, "dudx")),
    dnutdxID_(get_field_ordinal(meta, "dnutdx")),
    dualNodalVolumeID_(get_field_ordinal(meta, "dual_nodal_volume")),
    minDistanceID_(get_field_ordinal(meta, "minimum_distance_to_wall")),
    nDim_(meta.spatial_dimension())
{
}

void
SANuTildaNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

  nuTilda_ = fieldMgr.get_field<double>(nuTildaID_);
  density_ = fieldMgr.get_field<double>(densityID_);
  viscosity_ = fieldMgr.get_field<double>(viscosityID_);
  dudx_ = fieldMgr.get_field<double>(dudxID_);
  dnutdx_ = fieldMgr.get_field<double>(dnutdxID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
  minDistance_ = fieldMgr.get_field<double>(minDistanceID_);

  // Update turbulence model constants
  Cb1_ = realm.get_turb_model_constant(TM_saCb1);
  Cb2_ = realm.get_turb_model_constant(TM_saCb2);
  sigma_ = realm.get_turb_model_constant(TM_saSigma);
  kappa_ = realm.get_turb_model_constant(TM_kappa);
  Cv1_ = realm.get_turb_model_constant(TM_saCV1);
  Cw2_ = realm.get_turb_model_constant(TM_saCw2);
  Cw3_ = realm.get_turb_model_constant(TM_saCw3);
  Ct3_ = realm.get_turb_model_constant(TM_saCt3);
  Ct4_ = realm.get_turb_model_constant(TM_saCt4);

  // Derived constant
  Cw1_ = Cb1_ / (kappa_ * kappa_) + (1.0 + Cb2_) / sigma_;
}

KOKKOS_FUNCTION
void
SANuTildaNodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  using DblType = NodeKernelTraits::DblType;

  // Reference: https://turbmodels.larc.nasa.gov/spalart.html
  // Standard SA model (SA-noft2)

  const DblType nuTilda = nuTilda_.get(node, 0);
  const DblType rho = density_.get(node, 0);
  const DblType mu = viscosity_.get(node, 0);
  const DblType dVol = dualNodalVolume_.get(node, 0);
  const DblType d = minDistance_.get(node, 0);

  // Kinematic viscosity
  const DblType nu = mu / rho;

  // chi = nuTilda / nu
  const DblType chi = nuTilda / nu;
  const DblType chi3 = chi * chi * chi;

  // fv1 = chi^3 / (chi^3 + Cv1^3)
  const DblType Cv1_3 = Cv1_ * Cv1_ * Cv1_;
  const DblType fv1 = chi3 / (chi3 + Cv1_3);

  // fv2 = 1 - chi / (1 + chi * fv1)
  const DblType fv2 = 1.0 - chi / (1.0 + chi * fv1);

  // Vorticity magnitude
  DblType Omega = 0.0;
  for (int i = 0; i < nDim_; ++i) {
    for (int j = 0; j < nDim_; ++j) {
      const DblType Wij =
        0.5 * (dudx_.get(node, nDim_ * i + j) -
               dudx_.get(node, nDim_ * j + i));
      Omega += 2.0 * Wij * Wij;
    }
  }
  Omega = stk::math::sqrt(Omega);

  // STilda = Omega + nuTilda / (kappa^2 * d^2) * fv2
  // With clipping to prevent negative STilda
  const DblType Sbar = nuTilda * fv2 / (kappa_ * kappa_ * d * d);
  DblType STilda;
  if (Sbar >= -Cv1_ * Cv1_ * Omega) {
    STilda = Omega + Sbar;
  } else {
    const DblType Cv1_2 = Cv1_ * Cv1_;
    STilda = Omega + Omega * (Cv1_2 * Omega + Cv1_ * Cv1_ * Sbar) /
                       ((Cv1_2 - 2.0 * Cv1_2) * Omega - Sbar);
  }

  // Ensure STilda is positive (safety clipping)
  STilda = stk::math::max(STilda, 1.0e-16);

  // r = nuTilda / (STilda * kappa^2 * d^2)
  const DblType r_arg = nuTilda / (STilda * kappa_ * kappa_ * d * d);
  const DblType r = stk::math::min(r_arg, 10.0);

  // g = r + Cw2 * (r^6 - r)
  const DblType r6 = r * r * r * r * r * r;
  const DblType g = r + Cw2_ * (r6 - r);

  // fw = g * ((1 + Cw3^6) / (g^6 + Cw3^6))^(1/6)
  const DblType Cw3_6 = Cw3_ * Cw3_ * Cw3_ * Cw3_ * Cw3_ * Cw3_;
  const DblType g6 = g * g * g * g * g * g;
  const DblType fw = g * stk::math::pow((1.0 + Cw3_6) / (g6 + Cw3_6), 1.0 / 6.0);

  // dnutdx dot dnutdx (for Cb2 term)
  DblType dnutdx_sq = 0.0;
  for (int i = 0; i < nDim_; ++i) {
    const DblType dnutdxi = dnutdx_.get(node, i);
    dnutdx_sq += dnutdxi * dnutdxi;
  }

  // Production: Cb1 * STilda * nuTilda
  const DblType P_sa = Cb1_ * STilda * nuTilda;

  // Destruction: Cw1 * fw * (nuTilda / d)^2
  const DblType D_sa = Cw1_ * fw * (nuTilda / d) * (nuTilda / d);

  // Source from Cb2 term: (1/sigma) * Cb2 * dnutdx_i * dnutdx_i
  // (Note: this term arises from the non-conservative diffusion formulation
  //  and is added as a source. The conservative diffusion part 
  //  (1/sigma) * d/dx_j[(nu + nuTilda) * dnutdx_j] is handled by the
  //  edge solver with the effective viscosity.)
  const DblType S_cb2 = (1.0 / sigma_) * Cb2_ * dnutdx_sq;

  // RHS = (P_sa - D_sa + S_cb2) * rho * dVol
  // Linearize destruction term into LHS
  rhs(0) += (P_sa - D_sa + S_cb2) * rho * dVol;

  // LHS linearization:
  //   d(P_sa)/d(nuTilda) = Cb1 * STilda (approximate, ignoring STilda dependence)
  //   d(D_sa)/d(nuTilda) = 2 * Cw1 * fw * nuTilda / d^2
  const DblType lhsFac =
    (2.0 * Cw1_ * fw * nuTilda / (d * d) - Cb1_ * STilda) * rho * dVol;
  // Only add positive contributions to lhs for stability
  lhs(0, 0) += stk::math::max(lhsFac, 0.0);
}

} // namespace nalu
} // namespace sierra
