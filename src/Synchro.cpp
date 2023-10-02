#include "Synchro.h"

#define MIN(a, b) (a>b ? b : a)
#define M_DOUBLE_PI M_PI*2.0

static Eigen::Vector3d      red(1.0f, 0.0f, 0.0f);
static Eigen::Vector3d  darkred(0.545f, 0.0f, 0.0f);
static Eigen::Vector3d lightred(1.0f, 0.7f, 0.7f);
static Eigen::Vector3d    white(1.0f, 1.0f, 1.0f);
static Eigen::Vector3d darkgray(0.92f, 0.92f, 0.92f);

static Eigen::Vector3d road(0.7f, 0.7f, 0.5f);


static Eigen::Vector3d limegreen(0.19f, 0.8f, 0.19f);
static Eigen::Vector3d yellow(0.94f, 0.91f, 0.64f);
static Eigen::Vector3d darkorange(1.0f, 0.54f, 0.0f);
static Eigen::Vector3d blue(0.0f, 0.0f, 1.0f);

static Eigen::Vector3d tp_color = limegreen; //road;
static Eigen::Vector3d tn_color = limegreen; //red;
static Eigen::Vector3d fp_color = darkred;
static Eigen::Vector3d fn_color = blue;

Synchro::Synchro(YAML::Node &sample_data, bool graphic) {

  // build infos for graphics
  for (int level=0; ; level++) {
    auto cyl_s = std::string("cyl") + std::string(2 - MIN(2, std::to_string(level).length()), '0') + std::to_string(level);
    YAML::Node node = sample_data["general"][cyl_s.c_str()];
    if (!node) break;

    Info info;
    info.start_radius = node["min_radius"].as<float>();
    info.end_radius   = node["max_radius"].as<float>();
    info.steps_num    = node["steps_num"].as<int>();
    info.yaw_steps    = node["yaw_steps"].as<int>();
    info.z            = node["z_level"].as<float>();
    info.yaw_steps_half    = info.yaw_steps / 2;
    info.radius_step = ((info.end_radius - info.start_radius) * 100000 + 5) / info.steps_num / 100000.0f; // just a truncation
    info.yaw_res = M_DOUBLE_PI / info.yaw_steps;
    info.tot_cells = info.steps_num*info.yaw_steps;

    infos.push_back(info);
  }

  if (graphic) {
    createTriangs();
    update_geometry=false;
    spin_flag = true;
    workerThread = std::thread(&Synchro::update, this); 
  }
}

Synchro::~Synchro() {
  if (workerThread.joinable()) workerThread.join();
}

void Synchro::reset() {
  update_geometry = false;
  bt.reset();
}

void Synchro::update() {
  open3d::visualization::Visualizer vis;
  vis.CreateVisualizerWindow("Open3D", 2000, 2000);

  while(spin_flag) {
    // mu.lock(); // maybe unnecessary

    // if (!pointclouds_ptr.empty()) {
    //   vis.ClearGeometries();
    //   auto rr = pointclouds_ptr.front();
    //   // std::cout << "publishing cloud with " << rr->points_.size() << "\n";
    //   vis.AddGeometry(pointclouds_ptr.front(), false);
    //   pointclouds_ptr.pop();
    //   update_geometry=true;
    // }

    // if (!pointclouds_v_ptr.empty()) {
    //   vis.ClearGeometries();
    //   vis.AddGeometry(pointclouds_v_ptr.front(), false);
    //   pointclouds_v_ptr.pop();
    //   update_geometry=true;
    // }

    if (!polar_grids_ptr.empty()) {
      vis.AddGeometry(polar_grids_ptr.front(), false);
      polar_grids_ptr.pop();
      update_geometry=true;
    }

    if (update_geometry) {
      vis.UpdateGeometry();
      vis.UpdateRender();
      update_geometry = false;

      if (reset_view_flag) {    
        vis.ResetViewPoint(true);
        reset_view_flag = false;
      }

      // vis.GetViewControl().SetLookat(Eigen::Vector3d(0.0f, 0.0f, 0.0f));
      // vis.GetViewControl().SetConstantZNear(10.0f);
    }

    vis.PollEvents();
    // mu.unlock();

    // TODO: add time check (latency)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

}


void Synchro::resetViewFlag(bool reset_view_flag_) {
	reset_view_flag = reset_view_flag_;
}


void Synchro::addPointCloud(DataLoader &dl) {
  // pointclouds_ptr.push(dl.getPaintedCloud());
  pointclouds_ptr.push(dl.getUniformPaintedCloud());
}

void Synchro::addPointCloudVoxeled(DataLoader &dl, float voxel_size) {
  pointclouds_v_ptr.push(dl.getPaintedCloudVoxeled(voxel_size));
}

void Synchro::addPolarGrid(int level, std::vector<Cell> &grid) {
  updateTriang(level, grid);
  polar_grids_ptr.push(meshes[level]);
}

void Synchro::addPolarGridPred(int level, std::vector<Cell> &grid) {
  updateTriangPred(level, grid);
  polar_grids_ptr.push(meshes[level]);
}

void Synchro::delay(int vis_offset) {
  int rem = vis_offset-bt.elapsedTimeMs();
  while (rem>0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    rem -= 5;
  }
}

void Synchro::join() {
  spin_flag = false;
}


void Synchro::createTriangs() {

  for (int level=0; level<(int)infos.size(); level++) {

    auto info = infos[level];

    // initialize data structure
    auto mesh = std::make_shared<open3d::geometry::TriangleMesh>();
    mesh->vertices_.resize(4*info.tot_cells);
    mesh->triangles_.resize(4*info.tot_cells);
    mesh->vertex_colors_.resize(4*info.tot_cells);

    // handy variables to store intermediate values
    double sina, cosa, sina_, cosa_;
    float angle, r0, r1;
    int off=0;

    for (float r=info.start_radius; r<info.end_radius; r+=info.radius_step) {
      r0 = r; r1 = r + info.radius_step;

      for (float realangle=0; realangle<2*M_PI; realangle+=info.yaw_res) {

        // clip angle in range [0, 2M_PI]
        angle = realangle+M_PI;
            if (angle>=M_DOUBLE_PI) angle = angle - M_DOUBLE_PI;
        else if (angle<0.0f)         angle = M_DOUBLE_PI + angle;

        sina = sin(angle); sina_ = sin(angle+info.yaw_res);
        cosa = cos(angle); cosa_ = cos(angle+info.yaw_res);

        mesh-> vertices_[off] = Eigen::Vector3d(r0*sina,  r0*cosa,  info.z);
        mesh->triangles_[off] = Eigen::Vector3i(off,   off+1, off+2);
        off++;
        mesh-> vertices_[off] = Eigen::Vector3d(r1*sina,  r1*cosa,  info.z);
        mesh->triangles_[off] = Eigen::Vector3i(off+1, off,   off-1);
        off++;
        mesh-> vertices_[off] = Eigen::Vector3d(r0*sina_, r0*cosa_, info.z);
        mesh->triangles_[off] = Eigen::Vector3i(off-1, off,   off+1);
        off++;
        mesh-> vertices_[off] = Eigen::Vector3d(r1*sina_, r1*cosa_, info.z);
        mesh->triangles_[off] = Eigen::Vector3i(off,   off-1, off-2);
        off++;

      }
    }
    meshes.push_back(mesh);
  }
}

void Synchro::updateTriang(int level, std::vector<Cell> &grid) {
  Eigen::Vector3d color_;
  int idx, gtlabel;
  float label;
  int cont=0;
  int i=0; // TODO:remove dependency on this mechanism. just incremental, better
          // to do so, just create triangs in another order.
  auto info = infos[level];
  auto mesh = meshes[level];

  for (int row_idx = 0; row_idx<info.steps_num; row_idx++) {
    for (int yaw_idx = 0; yaw_idx<info.yaw_steps; yaw_idx++, i++) {

      idx = ((yaw_idx+info.yaw_steps/2)%info.yaw_steps)*info.steps_num + row_idx;
      label = grid[idx].predicted_label;
      gtlabel = grid[idx].label;

           if (gtlabel==UNKNOWN_CELL_LABEL) color_ = darkgray;
      else if (label>0) {
        if (label*gtlabel>0) color_ = white; //tp_color;
        else color_ = darkorange; //fp_color;
      }
      else {
        if (label*gtlabel>0) color_ = red; //tn_color;
        else color_ = blue; //fn_color;
      } 

      // else if (label>0) color_ = white; //limegreen;
      //   else color_ = red; //darkred;    
      // else if (gtlabel>0) color_ = white; //limegreen;
      //   else color_ = red; //darkred;                        

      // set the color for the cell
      mesh->vertex_colors_[cont ++] = (color_);
      mesh->vertex_colors_[cont ++] = (color_);
      mesh->vertex_colors_[cont ++] = (color_);
      mesh->vertex_colors_[cont ++] = (color_);
    }
  }

  //mesh->ComputeVertexNormals();

}

void Synchro::updateTriangPred(int level, std::vector<Cell> &grid) {
  Eigen::Vector3d color_;
  int idx, gtlabel;
  float label;
  int cont=0;
  int i=0; // TODO:remove dependency on this mechanism. just incremental, better
          // to do so, just create triangs in another order.
  auto info = infos[level];
  auto mesh = meshes[level];

  for (int row_idx = 0; row_idx<info.steps_num; row_idx++) {
    for (int yaw_idx = 0; yaw_idx<info.yaw_steps; yaw_idx++, i++) {

      idx = ((yaw_idx+info.yaw_steps/2)%info.yaw_steps)*info.steps_num + row_idx;
      label = grid[idx].predicted_label;
      gtlabel = grid[idx].label;

      if (gtlabel==UNKNOWN_CELL_LABEL || grid[idx].status==UNPREDICTABLE) color_ = darkgray;
      else {
        if (label>0) {
          color_ = white; //limegreen;
        }
        else {
          color_ = red; //lightred;
        }                         
      }

      // set the color for the cell
      mesh->vertex_colors_[cont ++] = (color_);
      mesh->vertex_colors_[cont ++] = (color_);
      mesh->vertex_colors_[cont ++] = (color_);
      mesh->vertex_colors_[cont ++] = (color_);
    }
  }

  //mesh->ComputeVertexNormals();

}

void Synchro::updateTriangGT(int level, std::vector<Cell> &grid) {
  Eigen::Vector3d color_;
  int idx, label;
  int cont=0;

  auto info = infos[level];
  auto mesh = meshes[level];

  for (int row_idx = 0; row_idx<info.steps_num; row_idx++) {
    for (int yaw_idx = 0; yaw_idx<info.yaw_steps; yaw_idx++) {

      idx = ((yaw_idx+info.yaw_steps/2)%info.yaw_steps)*info.steps_num + row_idx;
      label = grid[idx].label;

           if (label==UNKNOWN_CELL_LABEL) color_ = darkgray;
      else if (label==TRAV_CELL_LABEL)    color_ = tp_color;
      else                                color_ = tn_color;

      mesh->vertex_colors_[cont ++] = (color_);
      mesh->vertex_colors_[cont ++] = (color_);
      mesh->vertex_colors_[cont ++] = (color_);
      mesh->vertex_colors_[cont ++] = (color_);
    }
  }

  //mesh->ComputeVertexNormals();
}