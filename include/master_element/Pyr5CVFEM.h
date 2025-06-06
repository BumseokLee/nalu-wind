// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef Pyr5CVFEM_h
#define Pyr5CVFEM_h

#include <master_element/MasterElement.h>

#include <AlgTraits.h>

// NGP-based includes
#include "SimdInterface.h"
#include "KokkosInterface.h"

#include <cstdlib>
#include <stdexcept>
#include <array>

namespace stk {
struct topology;
}

namespace sierra {
namespace nalu {

struct ElementDescription;
class MasterElement;

// Pyramid 5 subcontrol volume
class PyrSCV : public MasterElement
{
public:
  using AlgTraits = AlgTraitsPyr5;
  using MasterElement::determinant;
  using MasterElement::grad_op;
  using MasterElement::shape_fcn;
  using MasterElement::shifted_grad_op;
  using MasterElement::shifted_shape_fcn;

  KOKKOS_FUNCTION
  PyrSCV();
  KOKKOS_FUNCTION virtual ~PyrSCV() {}

  KOKKOS_FUNCTION virtual const int* ipNodeMap(int ordinal = 0) const final;

  template <typename SCALAR, typename SHMEM>
  KOKKOS_FUNCTION void shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc);

  template <typename SCALAR, typename SHMEM>
  KOKKOS_FUNCTION void shifted_shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc);

  KOKKOS_FUNCTION virtual void determinant(
    const SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType*, DeviceShmem>& vol) override;

  virtual void determinant(
    const SharedMemView<double**>& coords,
    SharedMemView<double*>& vol) override;

  KOKKOS_FUNCTION void grad_op(
    const SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType***, DeviceShmem>& gradop,
    SharedMemView<DoubleType***, DeviceShmem>& deriv) override;

  KOKKOS_FUNCTION void shifted_grad_op(
    SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType***, DeviceShmem>& gradop,
    SharedMemView<DoubleType***, DeviceShmem>& deriv) override;

  KOKKOS_FUNCTION void Mij(
    SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType***, DeviceShmem>& metric,
    SharedMemView<DoubleType***, DeviceShmem>& deriv) override;

  void Mij(const double* coords, double* metric, double* deriv) override;

  void
  pyr_shape_fcn(const int npts, const double* par_coord, double* shape_fcn);

  void shifted_pyr_shape_fcn(
    const int npts, const double* par_coord, double* shape_fcn);

  virtual const double* integration_locations() const final { return intgLoc_; }
  virtual const double* integration_location_shift() const final
  {
    return intgLocShift_;
  }

protected:
  KOKKOS_FUNCTION virtual void
  shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc) override;
  virtual void shape_fcn(SharedMemView<double**, HostShmem>& shpfc) override;

  KOKKOS_FUNCTION virtual void
  shifted_shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc) override;
  virtual void
  shifted_shape_fcn(SharedMemView<double**, HostShmem>& shpfc) override;

private:
  static constexpr int nDim_ = AlgTraits::nDim_;
  static constexpr int nodesPerElement_ = AlgTraits::nodesPerElement_;
  static constexpr int numIntPoints_ = AlgTraits::numScvIp_;

  const int ipNodeMap_[AlgTraits::nodesPerElement_] = {0, 1, 2, 3, 4};

  const double one69r384 = 169.0 / 384.0;
  const double five77r3840 = 577.0 / 3840.0;
  const double seven73r1560 = 773.0 / 1560.0;

  const double intgLoc_[numIntPoints_ * nDim_] = {
    -one69r384, -one69r384, five77r3840, // vol 0
    one69r384,  -one69r384, five77r3840, // vol 1
    one69r384,  one69r384,  five77r3840, // vol 2
    -one69r384, one69r384,  five77r3840, // vol 3
    0.0,        0.0,        seven73r1560 // vol 4
  };

  const double intgLocShift_[15] = {
    -1.0, -1.0, 0.0, // vol 0
    1.0,  -1.0, 0.0, // vol 1
    1.0,  1.0,  0.0, // vol 2
    -1.0, 1.0,  0.0, // vol 3
    0.0,  0.0,  1.0  // vol 4
  };

  template <typename DBLTYPE, typename SHMEM>
  KOKKOS_INLINE_FUNCTION void determinant_scv(
    const SharedMemView<DBLTYPE**, SHMEM>& cordel,
    SharedMemView<DBLTYPE*, SHMEM>& vol);
};

// Pyramid 5 subcontrol surface
class PyrSCS : public MasterElement
{
public:
  using AlgTraits = AlgTraitsPyr5;
  using MasterElement::adjacentNodes;
  using MasterElement::determinant;
  using MasterElement::shape_fcn;
  using MasterElement::shifted_shape_fcn;

  KOKKOS_FUNCTION
  PyrSCS();
  KOKKOS_FUNCTION virtual ~PyrSCS() {}

  KOKKOS_FUNCTION virtual const int* ipNodeMap(int ordinal = 0) const final;

  template <typename SCALAR, typename SHMEM>
  KOKKOS_FUNCTION void shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc);

  template <typename SCALAR, typename SHMEM>
  KOKKOS_FUNCTION void shifted_shape_fcn(SharedMemView<SCALAR**, SHMEM>& shpfc);

  KOKKOS_FUNCTION virtual void determinant(
    const SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType**, DeviceShmem>& areav) override;

  virtual void determinant(
    const SharedMemView<double**>& coords,
    SharedMemView<double**>& areav) override;

  KOKKOS_FUNCTION void grad_op(
    const SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType***, DeviceShmem>& gradop,
    SharedMemView<DoubleType***, DeviceShmem>& deriv) override;

  void grad_op(
    const SharedMemView<double**>& coords,
    SharedMemView<double***>& gradop,
    SharedMemView<double***>& deriv) override;

  KOKKOS_FUNCTION void shifted_grad_op(
    SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType***, DeviceShmem>& gradop,
    SharedMemView<DoubleType***, DeviceShmem>& deriv) override;

  void pyr_derivative(const int npts, const double* intLoc, double* deriv);

  void
  shifted_pyr_derivative(const int npts, const double* intLoc, double* deriv);

  KOKKOS_FUNCTION void gij(
    const SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType***, DeviceShmem>& gupper,
    SharedMemView<DoubleType***, DeviceShmem>& glower,
    SharedMemView<DoubleType***, DeviceShmem>& deriv) override;

  KOKKOS_FUNCTION void Mij(
    SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType***, DeviceShmem>& metric,
    SharedMemView<DoubleType***, DeviceShmem>& deriv) override;

  void Mij(const double* coords, double* metric, double* deriv) override;

  KOKKOS_FUNCTION virtual const int* adjacentNodes() final;

  KOKKOS_FUNCTION virtual const int* scsIpEdgeOrd() final;

  void
  pyr_shape_fcn(const int npts, const double* par_coord, double* shape_fcn);

  void shifted_pyr_shape_fcn(
    const int npts, const double* par_coord, double* shape_fcn);

  void general_shape_fcn(
    const int numIp, const double* isoParCoord, double* shpfc) override
  {
    pyr_shape_fcn(numIp, isoParCoord, shpfc);
  }

  void sidePcoords_to_elemPcoords(
    const int& side_ordinal,
    const int& npoints,
    const double* side_pcoords,
    double* elem_pcoords) override;

  KOKKOS_FUNCTION int opposingNodes(const int ordinal, const int node) override;

  KOKKOS_FUNCTION int opposingFace(const int ordinal, const int node) override;

  KOKKOS_FUNCTION void face_grad_op(
    int face_ordinal,
    SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType***, DeviceShmem>& gradop,
    SharedMemView<DoubleType***, DeviceShmem>& deriv) final;

  KOKKOS_FUNCTION void shifted_face_grad_op(
    int face_ordinal,
    SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType***, DeviceShmem>& gradop,
    SharedMemView<DoubleType***, DeviceShmem>& deriv) final;

  void general_face_grad_op(
    const int face_ordinal,
    const double* isoParCoord,
    const double* coords,
    double* gradop,
    double* det_j,
    double* error) override;

  KOKKOS_FUNCTION const int* side_node_ordinals(int sideOrdinal) const final;

  double parametric_distance(const std::array<double, 3>& x);

  double isInElement(
    const double* elemNodalCoord,
    const double* pointCoord,
    double* isoParCoord) override;

  void interpolatePoint(
    const int& nComp,
    const double* isoParCoord,
    const double* field,
    double* result) override;

  virtual const double* integration_locations() const final { return intgLoc_; }
  virtual const double* integration_location_shift() const final
  {
    return intgLocShift_;
  }

protected:
  KOKKOS_FUNCTION virtual void
  shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc) override;
  virtual void shape_fcn(SharedMemView<double**, HostShmem>& shpfc) override;

  KOKKOS_FUNCTION virtual void
  shifted_shape_fcn(SharedMemView<DoubleType**, DeviceShmem>& shpfc) override;
  virtual void
  shifted_shape_fcn(SharedMemView<double**, HostShmem>& shpfc) override;

private:
  static constexpr int nDim_ = AlgTraits::nDim_;
  static constexpr int nodesPerElement_ = AlgTraits::nodesPerElement_;
  static constexpr int numIntPoints_ = AlgTraits::numScsIp_;

  const int sideNodeOrdinals_[16] = {
    0, 1, 4,   // ordinal 0
    1, 2, 4,   // ordinal 1
    2, 3, 4,   // ordinal 2
    0, 4, 3,   // ordinal 3
    0, 3, 2, 1 // ordinal 4
  };

  // define L/R mappings
  const int lrscv_[24] = {0, 1, 1, 2, 2, 3, 0, 3, 0, 4, 0, 4,
                          1, 4, 1, 4, 2, 4, 2, 4, 3, 4, 3, 4};

  // elem-edge map from ip
  const int scsIpEdgeOrd_[AlgTraits::numScsIp_] = {0, 1, 2, 3, 4, 4,
                                                   5, 5, 6, 6, 7, 7};

  // define opposing node
  // opposing node for node 4 is never uniquely defined: pick one
  const int oppNode_[20] = {
    // face 0; nodes 0,1,4
    3, 2, 2, -1,
    // face 1; nodes 1,2,4
    0, 3, 3, -1,
    // face 2; nodes 2,3,4
    1, 0, 0, -1,
    // face 3; nodes 0,4,3
    1, 1, 2, -1,
    // face 4; nodes 0,3,2,1
    4, 4, 4, 4};

#if 0
  // define opposing face
  // the 5th node maps to two opposing sub-faces, we pick one
  const int oppFace_[20] = {
  //face 0
  3, 1,  8, -1,
  //face 1
  0, 2, 10, -1,
  //face 2
  1, 3,  4, -1,
  //face 3
  0, 6,  2, -1,
  //face 4
  4, 10, 8,  6
  };
#endif

  const double twentynine63rd = 29.0 / 63.0;
  const double fortyone315th = 41.0 / 315.0;
  const double two9th = 2.0 / 9.0;
  const double thirteen45th = 13.0 / 45.0;
  const double seven18th = 7.0 / 18.0;

  const double intgLoc_[36] = {
    0.0,
    -twentynine63rd,
    fortyone315th, // surf 0  1->2
    twentynine63rd,
    0.0,
    fortyone315th, // surf 1  2->3
    0.0,
    twentynine63rd,
    fortyone315th, // surf 2  3->4
    -twentynine63rd,
    0.0,
    fortyone315th, // surf 3  1->4
    -two9th,
    -two9th,
    thirteen45th, // surf 4  1->5 inner
    -seven18th,
    -seven18th,
    seven18th, // surf 5  1->5 outer
    two9th,
    -two9th,
    thirteen45th, // surf 6  2->5 inner
    seven18th,
    -seven18th,
    seven18th, // surf 7  2->5 outer
    two9th,
    two9th,
    thirteen45th, // surf 8  3->5 inner
    seven18th,
    seven18th,
    seven18th, // surf 9  3->5 outer
    -two9th,
    two9th,
    thirteen45th, // surf 10  4->5 inner
    -seven18th,
    seven18th,
    seven18th // surf 11  4->5 outer
  };

  const double intgLocShift_[36] = {
    // shifted
    0.00,  -1.00, 0.00, // surf 1    1->2
    1.00,  0.00,  0.00, // surf 2    2->3
    0.00,  1.00,  0.00, // surf 3    3->4
    -1.00, 0.00,  0.00, // surf 4    1->4
    -0.50, -0.50, 0.50, // surf 5    1->5 I
    -0.50, -0.50, 0.50, // surf 6    1->5 O
    0.50,  -0.50, 0.50, // surf 7    2->5 I
    0.50,  -0.50, 0.50, // surf 8    2->5 O
    0.50,  0.50,  0.50, // surf 9    3->5 I
    0.50,  0.50,  0.50, // surf 10   3->5 O
    -0.50, 0.50,  0.50, // surf 11   4->5 I
    -0.50, 0.50,  0.50  // surf 12   4->5 O
  };

  const double seven36th = 7.0 / 36.0;
  const double twentynine36th = 29.0 / 36.0;
  const double five12th = 5.0 / 12.0;
  const double eleven18th = 11.0 / 18.0;

  const double intgExpFace_[48] = {
    // face 0, nodes 0,1,4: scs 0, 1, 2
    -five12th, -twentynine36th, seven36th, five12th, -twentynine36th, seven36th,
    0.0, -seven18th, eleven18th,
    // face 1, nodes 1,2,4, scs 0, 1, 2
    twentynine36th, -five12th, seven36th, twentynine36th, five12th, seven36th,
    seven18th, 0.0, eleven18th,
    // face 2, nodes 2,3,4, scs 0, 1, 2
    five12th, twentynine36th, seven36th, -five12th, twentynine36th, seven36th,
    0.00, seven18th, eleven18th,
    // face 3, nodes 0,4,3, scs 0, 1, 2
    -twentynine36th, -five12th, seven36th, -seven18th, 0.0, eleven18th,
    -twentynine36th, five12th, seven36th,
    // face 4, nodes 0,3,2,1, scs 0, 1, 2
    -0.5, -0.5, 0.0, -0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5, -0.5, 0.0};

  const int ipNodeMap_[16] = {
    // Face 0
    0, 1, 4,
    // Face 1
    1, 2, 4,
    // Face 2
    2, 3, 4,
    // Face 3
    0, 4, 3,
    // Face 4 (quad face)
    0, 3, 2, 1};

  double intgExpFaceShift_[48] = {0};

  template <typename DBLTYPE, typename SHMEM>
  KOKKOS_INLINE_FUNCTION void determinant_scs(
    const SharedMemView<DBLTYPE**, SHMEM>& cordel,
    SharedMemView<DBLTYPE**, SHMEM>& areav);
};

} // namespace nalu
} // namespace sierra

#endif
