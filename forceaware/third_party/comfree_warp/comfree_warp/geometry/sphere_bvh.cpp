// Exact nearest sphere in double precision; bounding nodes only prune losers.
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace {
constexpr int DIM = 3;
constexpr int LEAF_SIZE = 8;
struct Node {
  std::array<double, DIM> low, high, surface_low, surface_high;
  double radius;
  int start, stop, left = -1, right = -1;
};
struct Tree {
  std::vector<std::array<double, DIM>> centers;
  std::vector<double> radii;
  std::vector<int> order;
  std::vector<Node> nodes;

  Tree(const double* points, const double* sizes, int count)
      : centers(count), radii(sizes, sizes + count), order(count) {
    for (int i = 0; i < count; ++i)
      std::copy_n(points + DIM * i, DIM, centers[i].begin());
    std::iota(order.begin(), order.end(), 0);
    nodes.reserve(count);
    build(0, count);
  }

  Node bounds(int start, int stop) const {
    Node node{centers[order[start]], centers[order[start]],
              centers[order[start]], centers[order[start]], 0., start, stop};
    for (int i = start; i < stop; ++i) {
      node.radius = std::max(node.radius, radii[order[i]]);
      for (int axis = 0; axis < DIM; ++axis) {
        node.low[axis] = std::min(node.low[axis], centers[order[i]][axis]);
        node.high[axis] = std::max(node.high[axis], centers[order[i]][axis]);
        node.surface_low[axis] = std::min(node.surface_low[axis],
                                         centers[order[i]][axis] - radii[order[i]]);
        node.surface_high[axis] = std::max(node.surface_high[axis],
                                          centers[order[i]][axis] + radii[order[i]]);
      }
    }
    return node;
  }

  int build(int start, int stop) {
    const int id = static_cast<int>(nodes.size());
    nodes.push_back(bounds(start, stop));
    if (stop - start <= LEAF_SIZE) return id;
    int axis = 0;
    for (int d = 1; d < DIM; ++d)
      if (nodes[id].high[d] - nodes[id].low[d] >
          nodes[id].high[axis] - nodes[id].low[axis]) axis = d;
    const int middle = start + (stop - start) / 2;
    std::nth_element(order.begin() + start, order.begin() + middle, order.begin() + stop,
                     [&](int a, int b) { return centers[a][axis] < centers[b][axis]; });
    const int left = build(start, middle);
    const int right = build(middle, stop);
    nodes[id].left = left;
    nodes[id].right = right;
    return id;
  }

  double lower_bound(const double* point, int id) const {
    const auto& node = nodes[id];
    double squared = 0., surface_squared = 0.;
    double surface_inside = -std::numeric_limits<double>::infinity();
    for (int d = 0; d < DIM; ++d) {
      const double gap = std::max({node.low[d] - point[d], point[d] - node.high[d], 0.});
      squared += gap * gap;
      const double surface_gap = std::max(node.surface_low[d] - point[d],
                                          point[d] - node.surface_high[d]);
      surface_inside = std::max(surface_inside, surface_gap);
      surface_squared += std::max(surface_gap, 0.) * std::max(surface_gap, 0.);
    }
    // Combine conservative centre and surface AABB bounds.
    const double surface_bound = std::sqrt(surface_squared) + std::min(surface_inside, 0.);
    return std::max(std::sqrt(squared) - node.radius, surface_bound);
  }

  void leaf(const double* point, const Node& node, double& best, int& selected) const {
    for (int i = node.start; i < node.stop; ++i) {
      const int id = order[i];
      double squared = 0.;
      for (int d = 0; d < DIM; ++d) {
        const double gap = point[d] - centers[id][d];
        squared += gap * gap;
      }
      const double distance = std::sqrt(squared) - radii[id];
      if (distance < best || (distance == best && id < selected)) {
        best = distance;
        selected = id;
      }
    }
  }

  void visit(const double* point, int id, double& best, int& selected) const {
    const auto& node = nodes[id];
    if (node.left < 0) return leaf(point, node, best, selected);
    int first = node.left, second = node.right;
    double a = lower_bound(point, first), b = lower_bound(point, second);
    if (a > b) { std::swap(first, second); std::swap(a, b); }
    if (a <= best) visit(point, first, best, selected);
    if (b <= best) visit(point, second, best, selected);
  }

  bool ray_box(const double* point, const double* direction, const Node& node,
               double padding, double limit) const {
    double enter = 0., leave = limit;
    const double radius = std::max(0., node.radius + padding);
    for (int d = 0; d < DIM; ++d) {
      if (direction[d] == 0.) {
        if (point[d] < node.low[d] - radius || point[d] > node.high[d] + radius) return false;
        continue;
      }
      double a = (node.low[d] - radius - point[d]) / direction[d];
      double b = (node.high[d] + radius - point[d]) / direction[d];
      if (a > b) std::swap(a, b);
      enter = std::max(enter, a);
      leave = std::min(leave, b);
    }
    return enter <= leave;
  }

  void ray_leaf(const double* point, const double* direction, const Node& node,
                double padding, double limit,
                std::vector<std::array<double, 2>>& intervals) const {
    for (int i = node.start; i < node.stop; ++i) {
      const int id = order[i];
      double parallel = 0., squared = 0.;
      for (int d = 0; d < DIM; ++d) {
        const double delta = point[d] - centers[id][d];
        parallel += delta * direction[d];
        squared += delta * delta;
      }
      const double radius = std::max(0., radii[id] + padding);
      const double discriminant = parallel * parallel - squared + radius * radius;
      if (discriminant < 0.) continue;
      const double root = std::sqrt(discriminant);
      const double enter = -parallel - root, leave = -parallel + root;
      if (leave >= 0. && enter <= limit) intervals.push_back({enter, leave});
    }
  }

  void ray_visit(const double* point, const double* direction, int id, double padding,
                 double limit, std::vector<std::array<double, 2>>& intervals) const {
    const auto& node = nodes[id];
    if (!ray_box(point, direction, node, padding, limit)) return;
    if (node.left < 0) return ray_leaf(point, direction, node, padding, limit, intervals);
    ray_visit(point, direction, node.left, padding, limit, intervals);
    ray_visit(point, direction, node.right, padding, limit, intervals);
  }

  double retreat(const double* points, int count, const double* direction,
                 double padding, double limit) const {
    std::vector<std::array<double, 2>> intervals;
    for (int i = 0; i < count; ++i)
      ray_visit(points + DIM * i, direction, 0, padding, limit, intervals);
    std::sort(intervals.begin(), intervals.end());
    double result = 0.;
    for (const auto& interval : intervals) {
      if (interval[0] > result) break;
      result = std::max(result, interval[1]);
    }
    return std::nextafter(result, std::numeric_limits<double>::infinity());
  }
};
}

extern "C" {
void* sphere_bvh_create(const double* centers, const double* radii, int count) {
  try { return new Tree(centers, radii, count); }
  catch (...) { return nullptr; }
}
void sphere_bvh_destroy(void* tree) { delete static_cast<Tree*>(tree); }
void sphere_bvh_query(void* handle, const double* points, int count, int* ids, int threads) {
  const auto* tree = static_cast<const Tree*>(handle);
  #pragma omp parallel for num_threads(threads) if(threads > 1) schedule(static)
  for (int i = 0; i < count; ++i) {
    double best = std::numeric_limits<double>::infinity();
    int selected = static_cast<int>(tree->radii.size());
    tree->visit(points + DIM * i, 0, best, selected);
    ids[i] = selected;
  }
}
double sphere_bvh_retreat(void* handle, const double* points, int count,
                          const double* direction, double padding, double limit) {
  try { return static_cast<Tree*>(handle)->retreat(points, count, direction, padding, limit); }
  catch (...) { return std::numeric_limits<double>::quiet_NaN(); }
}
}
