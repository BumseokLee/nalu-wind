#ifndef TIOGABLOCK_H
#define TIOGABLOCK_H

#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>

#include "overset/TiogaOptions.h"
#include "overset/OversetFieldData.h"
#include "overset/OversetNGP.h"
#include "yaml-cpp/yaml.h"

#include <vector>
#include <memory>
#include <string>

namespace TIOGA {
class tioga;
}

namespace tioga_nalu {

typedef stk::mesh::Field<double, stk::mesh::Cartesian> VectorFieldType;
typedef stk::mesh::Field<double> ScalarFieldType;
typedef stk::mesh::Field<int> ScalarIntFieldType;

/** Data representing an unstructured mesh block
 */
struct NgpTiogaBlock
{
  /** Number of cell types supported by TIOGA
   *
   *  Currently it supports hex, tet, wedge, and pyramids
   */
  static constexpr int max_vertex_types = 4;

  //! Coordinates of the nodes for this mesh (size = 3 * num_nodes)
  OversetArrayType<double*> xyz_;

  //! Nodal resolutions used for resolution check (size = num_nodes)
  OversetArrayType<double*> node_res_;

  //! Cell resolutions used for resolution checks (size = num_elements)
  OversetArrayType<double*> cell_res_;

  //! IBLANK array populated after overset connectivity (size = num_nodes)
  OversetArrayType<int*> iblank_;

  //! IBLANK array populated after overset connectivity (size = num_nodes)
  OversetArrayType<int*> iblank_cell_;

  /** Indices of the nodes that define the wall boundaries
   *
   *  The indices are 1-based (Fortran style)
   */
  OversetArrayType<int*> wallIDs_;

  /** Indices of the nodes that define the overset boundaries
   *
   *  The indices are 1-based (Fortran style)
   */
  OversetArrayType<int*> ovsetIDs_;

  /** Array indicating the number of vertices in the element topologies
   *
   *  The number of vertices indicate the element type, i.e., `8 = hex, 6 =
   *  wedge, 5 = pyramid, 4 = tet`. No other values are allowed.
   *
   *  For a mesh containing solely hex elements, this array just contains one
   *  entry: `{8}`.
   */
  OversetArrayType<int*> num_verts_;

  /** Array indicating the number of elements for each element topology.
   *
   *  This array is the same size as `num_verts_`.
   */
  OversetArrayType<int*> num_cells_;

  /** Element connectivity information
   *
   *  Pointers to arrays containing connectivity information for the elements.
   *  Ony num_vertex_types entries are filled. Each array has (num_vertices *
   *  num_cells) entries for individual topologies.
   */
  OversetArrayType<int*> connect_[max_vertex_types];

  //! TIOGA index lookup using local entity index. TIOGA uses 1-based indexing,
  //! so the first entry (node/element) has the index set to 1. This must be
  //! taken into account when dereferencing entries in TIOGA data arrays.
  OversetArrayType<int*> eid_map_;

  //! The global STK identifier for this node. Used to identify shared nodes
  //! across domain partitions.
  OversetArrayType<stk::mesh::EntityId*> node_gid_;

  //! The global STK identifier for an element. Used to identify donors on a
  //! different MPI partition at the receptor MPI ranks. Not necessary if STK
  //! ghosting is not used.
  OversetArrayType<stk::mesh::EntityId*> cell_gid_;

  //! Solution array used to exchange data between meshes through TIOGA
  OversetArrayType<double*> qsol_;
};

/**
 * Interface to convert STK Mesh Part(s) to TIOGA blocks.
 *
 * This class provides a mapping between STK mesh parts and the concept of a
 * TIOGA mesh block. Each TIOGA mesh block is determined by a unique body tag
 * and requires information of all the nodes and elements comprising the mesh
 * block (within this MPI rank). TIOGA determines the global mesh information by
 * looking up the unique body tag across all MPI ranks. TIOGA requires
 * information regarding the volume mesh as well as the wall and overset
 * surfaces.
 *
 * TIOGA communicates overset connectivity via IBLANK (node) and IBLANK_CELL
 * (element) masking arrays that have flags indicating whether a node/element is
 * a hole, fringe, or a field point.
 */
class TiogaBlock
{
public:
  TiogaBlock(
    stk::mesh::MetaData&,
    stk::mesh::BulkData&,
    TiogaOptions&,
    const YAML::Node&,
    const std::string,
    const int);

  ~TiogaBlock();

  /** Setup block structure information (steps before mesh creation)
   */
  void setup(stk::mesh::PartVector&);

  /** Initialize mesh data structure (steps after mesh creation)
   */
  void initialize();

  /** Update coordinates upon mesh motion
   *
   *  Update the coordinates sent to TIOGA from STK. This assumes that the mesh
   *  connectivity information itself does not change, i.e., no refinement, etc.
   *
   *  Updates to mesh connectivity information will require a call to
   *  TiogaBlock::update_connectivity() instead.
   */
  void update_coords();

  /** Perform full update including connectivity
   *
   */
  void update_connectivity();

  /** Update cell volumes
   *
   */
  void update_element_volumes();

  /** Adjust resolutions of mandatory fringe entities
   */
  void adjust_cell_resolutions();

  void adjust_node_resolutions();

  /** Register this block with TIOGA
   *
   *  Wrapper method to handle mesh block registration using TIOGA API. In
   *  addition to registering the mesh block, it will also provide IBLANK_CELL
   *  data structure for TIOGA to populate.
   *
   *  The interface also allows registration of user-defined node and cell
   *  resolution information that enables the user to force a certain type of
   *  overset holecutting that overrides the default TIOGA behavior of selecting
   *  donor and receptor points based on local cell volume.
   */
  void register_block(TIOGA::tioga&);

  /** Update iblanks after connectivity updates
   */
  void update_iblanks();
  /** Update fringe and hole node vectors
   */
  void update_fringe_and_hole_nodes(
    std::vector<stk::mesh::Entity>&, std::vector<stk::mesh::Entity>&);
  /** Update the Tioga view of iblanks prior to donor-to-receptor interpolation
   */
  void update_tioga_iblanks();

  /** Update element iblanks after connectivity updates
   */
  void update_iblank_cell();

  /** Determine the custom ghosting elements for this mesh block
   *
   *  Calls the TIOGA API and populates the elements that need ghosting to other
   *  MPI ranks.
   *
   *  @param tg Reference to TIOGA API object (provided by TiogaSTKIface).
   *  @param egvec List of {donorElement, receptorMPIRank} pairs to be populated
   */
  void get_donor_info(TIOGA::tioga&, stk::mesh::EntityProcVec&);

  void register_solution(
    TIOGA::tioga&,
    const std::vector<sierra::nalu::OversetFieldData>&,
    const int);

  void register_solution(TIOGA::tioga&, const sierra::nalu::OversetFieldData&);

  void update_solution(const std::vector<sierra::nalu::OversetFieldData>&);

  void update_solution(const sierra::nalu::OversetFieldData&);

  //! Return the block name for this mesh
  const std::string& block_name() const { return block_name_; }

private:
  TiogaBlock() = delete;
  TiogaBlock(const TiogaBlock&) = delete;

  /** Process the YAML node to gather all user inputs
   */
  void load(const YAML::Node&);

  /** Convenience function to process part names and populate a PartVector
   */
  inline void
  names_to_parts(const std::vector<std::string>&, stk::mesh::PartVector&);

  /**
   * Extract nodes from all parts to send to TIOGA
   */
  void process_nodes();

  /** Determine the local indices (into the TIOGA mesh block data structure) of
   * all the wall boundary nodes.
   */
  void process_wallbc();

  /** Determine the local indices (into the TIOGA mesh block data structure) of
   *  all the overset boundary nodes.
   */
  void process_ovsetbc();

  /** Generate the element data structure and connectivity information to send
   * to TIOGA
   */
  void process_elements();

  /** Reset iblank data with moving mesh applications
   *
   */
  void reset_iblank_data();

  /** Print summary of mesh blocks
   */
  void print_summary();

  /** Return a selector for accessing nodes for use with TIOGA API
   *
   *  This is necessary to avoid selecting nodes that are ghosted and can cause
   *  issues with the hole-cutting logic.
   */
  stk::mesh::Selector get_node_selector(stk::mesh::PartVector&);

  /** Return a selector for accessing elements for use with TIOGA API
   */
  stk::mesh::Selector get_elem_selector(stk::mesh::PartVector&);

  //! Reference to the STK Mesh MetaData object
  stk::mesh::MetaData& meta_;

  //! Reference to the STK Mesh BulkData object
  stk::mesh::BulkData& bulk_;

  //! Options controlling TIOGA holecutting
  TiogaOptions tiogaOpts_;

  //! Part names for the nodes for this mesh block
  std::vector<std::string> blkNames_;

  //! Part names for the wall boundaries
  std::vector<std::string> wallNames_;

  //! Part names for the overset boundaries
  std::vector<std::string> ovsetNames_;

  //! Mesh parts for the nodes
  stk::mesh::PartVector blkParts_;

  //! Wall BC parts
  stk::mesh::PartVector wallParts_;

  //! Overset BC parts
  stk::mesh::PartVector ovsetParts_;

  //! Data representing this unstructured block
  NgpTiogaBlock bdata_;

  /** Connectivity map.
   *
   *  This map holds the number of elements present per topology type (npe ->
   *  num_elements).
   */
  std::map<int, int> conn_map_;

  /** Tioga connectivity data structure
   *
   */
  int** tioga_conn_{nullptr};

  //! Receptor information for this mesh block
  std::vector<int> receptor_info_;

  //! Name of coordinates Field
  std::string coords_name_;

  //! Dimensionality of the mesh
  int ndim_;

  //! Global mesh tag identifier
  int meshtag_;

  //! Name of this overset mesh block
  std::string block_name_;

  //! Number of nodes for this mesh
  int num_nodes_{0};

  //! Number of wall BC nodes (in this processor)
  int num_wallbc_{0};

  //! Number of overset BC nodes (in this processor)
  int num_ovsetbc_{0};

  //! Flag to check if we are are already initialized
  bool is_init_{true};

public:
  // Accessors

  //! STK Global ID for all the nodes comprising this mesh block
  inline auto node_id_map() const -> decltype(bdata_.node_gid_.h_view)
  {
    return bdata_.node_gid_.h_view;
  }

  //! STK Global ID for all the elements comprising this mesh block
  inline auto elem_id_map() const -> decltype(bdata_.cell_gid_.h_view)
  {
    return bdata_.cell_gid_.h_view;
  }

  //! IBLANK mask indicating whether the element is active or inactive
  inline auto iblank_cell() const -> decltype(bdata_.iblank_cell_.h_view)
  {
    return bdata_.iblank_cell_.h_view;
  }
};

} // namespace tioga_nalu

#endif /* TIOGABLOCK_H */
