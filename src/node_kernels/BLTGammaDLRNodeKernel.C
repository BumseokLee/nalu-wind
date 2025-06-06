/*------------------------------------------------------------------------*/
/*  Copyright 2017 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "node_kernels/BLTGammaDLRNodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "SimdInterface.h"
#include "utils/StkHelpers.h"
#include "NaluEnv.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

BLTGammaDLRNodeKernel::BLTGammaDLRNodeKernel(const stk::mesh::MetaData& meta)
  : NGPNodeKernel<BLTGammaDLRNodeKernel>(),
    tkeID_(get_field_ordinal(meta, "turbulent_ke")),
    sdrID_(get_field_ordinal(meta, "specific_dissipation_rate")),
    densityID_(get_field_ordinal(meta, "density")),
    viscID_(get_field_ordinal(meta, "viscosity")),
    tviscID_(get_field_ordinal(meta, "turbulent_viscosity")),
    dpdxID_(get_field_ordinal(meta, "dpdx")),
    velocityID_(get_field_ordinal(meta, "velocity")),
    pressureID_(get_field_ordinal(meta, "pressure")),
    dudxID_(get_field_ordinal(meta, "dudx")),
    minDID_(get_field_ordinal(meta, "minimum_distance_to_wall")),
    dwalldistdxID_(get_field_ordinal(meta, "dwalldistdx")),
    dnDotVdxID_(get_field_ordinal(meta, "dnDotVdx")),
    dualNodalVolumeID_(get_field_ordinal(meta, "dual_nodal_volume")),
    gamintID_(get_field_ordinal(meta, "gamma_transition")),
    nDim_(meta.spatial_dimension())
{
}

void
BLTGammaDLRNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

  tke_ = fieldMgr.get_field<double>(tkeID_);
  sdr_ = fieldMgr.get_field<double>(sdrID_);
  density_ = fieldMgr.get_field<double>(densityID_);
  visc_ = fieldMgr.get_field<double>(viscID_);
  tvisc_ = fieldMgr.get_field<double>(tviscID_);
  dpdx_ = fieldMgr.get_field<double>(dpdxID_);
  velocity_ = fieldMgr.get_field<double>(velocityID_);
  pressure_ = fieldMgr.get_field<double>(pressureID_);
  dudx_ = fieldMgr.get_field<double>(dudxID_);
  minD_ = fieldMgr.get_field<double>(minDID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
  gamint_ = fieldMgr.get_field<double>(gamintID_);
  fsti_ = realm.get_turb_model_constant(TM_fsti);
}

KOKKOS_FUNCTION
void
BLTGammaDLRNodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  using DblType = NodeKernelTraits::DblType;

  const DblType tke = tke_.get(node, 0);
  const DblType sdr = sdr_.get(node, 0);
  const DblType gamint = gamint_.get(node, 0);

  const DblType density = density_.get(node, 0);
  const DblType visc = visc_.get(node, 0);
  const DblType tvisc = tvisc_.get(node, 0);
  const DblType minD = minD_.get(node, 0);
  const DblType dVol = dualNodalVolume_.get(node, 0);

  // Gas constants of air
  const DblType pinf = 101325.0;
  const DblType Rair = 8314.4621 / 28.96;
  const DblType Tinf = pinf / (Rair * density);
  const DblType kappa = 1.4;
  const DblType km1 = 0.4;

  const DblType pressure =
    pressure_.get(node, 0) + pinf; // add reference pressure

  // constants for the source terms
  const DblType flength = 14.0;
  const DblType caTwo = 0.06;
  const DblType ceTwo = 50.0;

  // Hard-coding now
  const DblType Uinf = 34.1;
  const DblType Minf = 0.1;

  DblType dvnn = 0.0;
  DblType TuL = 0.0;
  DblType lamda0L = 0.0;

  DblType sijMag = 0.0;
  DblType vortMag = 0.0;

  for (int i = 0; i < nDim_; ++i) {
    for (int j = 0; j < nDim_; ++j) {
      const double duidxj = dudx_.get(node, nDim_ * i + j);
      const double dujdxi = dudx_.get(node, nDim_ * j + i);

      const double rateOfStrain = 0.5 * (duidxj + dujdxi);
      const double vortTensor = 0.5 * (duidxj - dujdxi);
      sijMag += rateOfStrain * rateOfStrain;
      vortMag += vortTensor * vortTensor;
    }
  }

  sijMag = stk::math::sqrt(2.0 * sijMag);
  vortMag = stk::math::sqrt(2.0 * vortMag);

  const DblType Rev = density * minD * minD * sijMag / visc;

  if (fsti_ > 0.0) {
    TuL = fsti_; // const. Tu from yaml
  } else {
    TuL = stk::math::min(
      100.0 * stk::math::sqrt(2.0 / 3.0 * tke) / sdr / (minD + 1.0e-10),
      100.0); // local Tu
  }

  // Compute Boundary Layer Edge Values
  DblType var1 = 1 + km1 / 2 * Minf * Minf;
  DblType var2 = stk::math::pow(pressure / pinf, km1 / kappa);

  const DblType Me = stk::math::sqrt((var1 / var2 - 1) * 2 / km1);
  const DblType rhoe = density * stk::math::pow(pressure / pinf, 1 / kappa);
  const DblType Te = Tinf * stk::math::pow(pressure / pinf, km1 / kappa);
  const DblType Ue = stk::math::sqrt(
    Uinf * Uinf + 2 * kappa / km1 *
                    (1 - stk::math::pow(pressure / pinf, km1 / kappa)) * pinf /
                    density);

  //// Bumseok: Test both options for Sutherland law
  // Option 1. Anderson
  //  const DblType Su = 110.4;
  //  const DblType Tref = 288.16;
  //  const DblType muref = 1.7894e-5;

  // Option 2. Initial conditions
  const DblType Su = 110.4;
  const DblType Tref = Tinf;
  const DblType muref = visc;

  const DblType nue =
    muref / rhoe * stk::math::pow(Te / Tref, 1.5) * (Tref + Su) / (Te + Su);

  // Compute dUedx:
  // Bumseok: need to account for the velocity for moving objects
  DblType u_mag = 0.0;
  for (int i = 0; i < nDim_; ++i)
    u_mag += velocity_.get(node, i) * velocity_.get(node, i);
  u_mag = stk::math::sqrt(u_mag);

  DblType dUeds = 0.0;
  DblType dUedx = 0.0;
  for (int i = 0; i < nDim_; ++i) {
    dUedx = -1 / (Ue * density) *
            stk::math::pow(pressure / pinf, -1.0 / kappa) * dpdx_.get(node, i);
    dUeds += velocity_.get(node, i) / u_mag * dUedx;
  }

  // Compute the pressure gradient parameter lambda_t iteratively
  // The original AHD paper suggests that lambda_t should be between -0.068253
  // and 0.1
  const DblType lambda_t_min = -0.068253;
  const DblType lambda_t_max = 0.1;

  // Initial guess for lambda_t
  DblType lambda_t = 0.0;
  // DblType lambda_t = (lambda_t_min + lambda_t_max)/2;
  DblType theta_t = 0.0;
  DblType theta_t_old = 0.0;
  DblType h12 = 0.0;
  DblType pi_shape = 0.0;

  int iter = 0;
  int iter_max = 15; // Recomended one 15
  DblType resid_theta = 1e10;

  // Iteratively solve for theta_t and lambda_t: fixed-point method
  while (iter < iter_max) {
    iter++;
    theta_t_old = theta_t;

    if (lambda_t >= 0.0) {
      h12 = 4.02923 - stk::math::sqrt(
                        -8838.4 * stk::math::pow(lambda_t, 4) +
                        1105.1 * stk::math::pow(lambda_t, 3) -
                        67.962 * stk::math::pow(lambda_t, 2) +
                        17.574 * lambda_t + 2.0593);
    } else {
      // This is added by DLR, not in the original AHD
      h12 = 2.072 + 0.0731 / (lambda_t + 0.14);
    }

    pi_shape = 0.071665 * stk::math::pow(h12, 3) -
               0.73186 * stk::math::pow(h12, 2) + 4.2563 * h12 - 5.1743;

    theta_t = Rev / pi_shape * nue / Ue;

    lambda_t = stk::math::pow(Rev / pi_shape, 2) * nue / (Ue * Ue) * dUeds;

    lambda_t =
      stk::math::max(stk::math::min(lambda_t, lambda_t_max), lambda_t_min);

    // Compute residual
    resid_theta = stk::math::abs(theta_t - theta_t_old);

    if (resid_theta <= 1e-10) {
      break; // Convergence achieved
    }
  }

  const DblType Ret = -(177 * Me * Me - 22 * Me + 210) *
                      stk::math::log((7 * Me + 4.8) * TuL / 100.0) *
                      stk::math::exp((5 * Me + 27) * lambda_t);

  const DblType fonset1 = Rev / pi_shape / Ret;
  const DblType fonset2 =
    stk::math::min(stk::math::max(fonset1, stk::math::pow(fonset1, 4)), 2.0);
  const DblType rt = tvisc / visc; // mut/mu
  const DblType fonset3 =
    stk::math::max(1.0 - stk::math::pow(rt / 2.0, 3), 0.0);
  const DblType fonset = stk::math::max(fonset2 - fonset3, 0.0);
  const DblType fturb = stk::math::exp(-stk::math::pow(rt / 4.0, 4));

  const DblType Pgamma = flength * density * sijMag * fonset * (1.0 - gamint);
  const DblType Dgamma =
    caTwo * density * vortMag * fturb * gamint * (ceTwo * gamint - 1.0);

  const DblType PgammaDir = -flength * density * sijMag * fonset;
  const DblType DgammaDir =
    caTwo * density * vortMag * fturb * (2.0 * ceTwo * gamint - 1.0);

  rhs(0) += (Pgamma - Dgamma) * dVol;
  lhs(0, 0) += (DgammaDir - PgammaDir) * dVol;
}

} // namespace nalu
} // namespace sierra
