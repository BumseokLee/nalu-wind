// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/TKESSTBLTDLRNodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "SimdInterface.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

TKESSTBLTDLRNodeKernel::TKESSTBLTDLRNodeKernel(
  const stk::mesh::MetaData& meta)
  : NGPNodeKernel<TKESSTBLTDLRNodeKernel>(),
    tkeID_(get_field_ordinal(meta, "turbulent_ke")),
    sdrID_(get_field_ordinal(meta, "specific_dissipation_rate")),
    gamintID_(get_field_ordinal(meta, "gamma_transition")),
    densityID_(get_field_ordinal(meta, "density")),
    viscID_(get_field_ordinal(meta, "viscosity")),
    tviscID_(get_field_ordinal(meta, "turbulent_viscosity")),
    dudxID_(get_field_ordinal(meta, "dudx")),
    dualNodalVolumeID_(get_field_ordinal(meta, "dual_nodal_volume")),
    nDim_(meta.spatial_dimension())
{
}

void
TKESSTBLTDLRNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

  tke_ = fieldMgr.get_field<double>(tkeID_);
  sdr_ = fieldMgr.get_field<double>(sdrID_);
  density_ = fieldMgr.get_field<double>(densityID_);
  tvisc_ = fieldMgr.get_field<double>(tviscID_);
  dudx_ = fieldMgr.get_field<double>(dudxID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
  gamint_ = fieldMgr.get_field<double>(gamintID_);
  visc_ = fieldMgr.get_field<double>(viscID_);

  // Update turbulence model constants
  betaStar_ = realm.get_turb_model_constant(TM_betaStar);
  tkeProdLimitRatio_ = realm.get_turb_model_constant(TM_tkeProdLimitRatio);
  tkeAmb_ = realm.get_turb_model_constant(TM_tkeAmb);
  sdrAmb_ = realm.get_turb_model_constant(TM_sdrAmb);
}

KOKKOS_FUNCTION
void
TKESSTBLTDLRNodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  using DblType = NodeKernelTraits::DblType;

  // See https://turbmodels.larc.nasa.gov/sst.html for details

  const DblType tke = tke_.get(node, 0);
  const DblType sdr = sdr_.get(node, 0);
  const DblType density = density_.get(node, 0);
  const DblType tvisc = tvisc_.get(node, 0);
  const DblType dVol = dualNodalVolume_.get(node, 0);

  const DblType gamint = gamint_.get(node, 0);
  const DblType visc = visc_.get(node, 0);

  DblType sijMag = 1.0e-16;
  DblType vortMag = 1.0e-16;

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

  // Bumseok: not clear which one is correct: need to test all the options

  //// Option 1 no production the limiter
  //const DblType Pk = gamint * tvisc * sijMag * sijMag;

  //// Option 1-1: Kato-Launder, no production the limiter like M15
  //const DblType Pk = gamint * tvisc * vortMag * sijMag;

  //// Option 2: with the limiter, what I tested
  DblType Pk = gamint * tvisc * sijMag * sijMag;
  Pk = stk::math::min(Pk, tkeProdLimitRatio_ * betaStar_ * density * sdr * tke);

  // Option 3: similar to LM transition model
  //DblType Pk = tvisc * sijMag * sijMag;
  //Pk = gamint * stk::math::min(Pk, tkeProdLimitRatio_ * betaStar_ * density * sdr * tke);

  const DblType Dk =
    betaStar_ * density * sdr * tke * stk::math::min(stk::math::max(gamint, 0.1), 1.0);

  // SUST source term
  const DblType Dkamb = betaStar_ * density * sdrAmb_ * tkeAmb_;

  rhs(0) += (Pk - Dk + Dkamb) * dVol;
  lhs(0, 0) += betaStar_ * density * sdr * stk::math::min(stk::math::max(gamint, 0.1), 1.0) * dVol;
}

} // namespace nalu
} // namespace sierra
