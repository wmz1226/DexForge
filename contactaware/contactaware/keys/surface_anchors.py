"""Shared stable anchors as optimization variables constrained to the object surface."""

from dataclasses import dataclass, replace
import numpy as np

from contactaware.initialization.multi_contact_keys import MultiContactProblem, ConstraintRowBlock

XYZ_DIM = 3


class SurfaceAnchorProblem(MultiContactProblem):
    @property
    def coordinate_description(self):
        return ("joint key qpos and independent shared surface-anchor coordinates; "
                "original segment identities and representative keys retained")

    def __init__(self, base):
        super().__init__(hand=base.hand, cfg=base.cfg, settings=base.settings,
            schedule=base.schedule, query_ids=base.query_ids, initial=base.initial,
            references=base.references, joints_obj=base.joints_obj, geometry=base.geometry,
            center=base.center, kinematics=base.kinematics, score=base.score,
            palm_terms=base.palm_terms, shape_terms=base.shape_terms, finger_ids=base.finger_ids,
            distribution_weights=base.distribution_weights, pre_contact=base.pre_contact,
            topology_reference=base.topology_reference)
        self.state_dimension = base.dimension
        self.initial_anchors = base.constraint_evaluation(np.zeros(base.dimension)).shared[0].copy()
        self.anchor_dimension = self.initial_anchors.size
        self.dimension = self.state_dimension + self.anchor_dimension
        self._base_geometry, self._surface_geometry = None, None

    def query_surfaces(self, robots):
        points = np.concatenate([robot.points for robot in robots])
        values, derivatives = self.geometry.distance_derivatives(points)
        sizes = np.cumsum([len(robot.points) for robot in robots])[:-1]
        return tuple(zip(np.split(values, sizes), derivatives.split(sizes)))

    def states(self, vector):
        return super().states(np.asarray(vector)[:self.state_dimension])

    def bounds(self):
        lower, upper = super().bounds()
        return np.r_[lower, np.full(self.anchor_dimension,-np.inf)], np.r_[upper,np.full(self.anchor_dimension,np.inf)]

    def trust(self):
        return np.r_[super().trust(),np.full(self.anchor_dimension,self.cfg.base_translation_trust_m/self.radius)]

    def anchor_coordinates(self, vector):
        return self.initial_anchors + np.asarray(vector)[self.state_dimension:].reshape(-1,XYZ_DIM)*self.radius

    def shared_geometry(self, vector, robots, surfaces):
        points = self.anchor_coordinates(vector)
        values, derivatives = self.geometry(points)
        rows = np.zeros((len(points),XYZ_DIM,self.dimension))
        rows[:,:,self.state_dimension:] = self.radius*np.eye(self.anchor_dimension).reshape(len(points),XYZ_DIM,-1)
        normal_rows = np.einsum('nij,njk->nik',derivatives[:,4:7],rows)
        return points,values[:,4:7],rows,normal_rows

    def constraint_evaluation(self, vector):
        geometry = super().constraint_evaluation(vector)
        if geometry is self._base_geometry:
            return self._surface_geometry
        points,_,point_rows,_ = geometry.shared
        values,derivatives = self.geometry(points)
        phi = values[:,0]/self.radius
        rows = np.einsum('ni,nij->nj',derivatives[:,0],point_rows)/self.radius
        group = int(geometry.groups.max())+1
        result = replace(geometry,constraints=np.r_[geometry.constraints,phi,-phi],
            row_factories=(*geometry.row_factories,
                ConstraintRowBlock(len(phi),self.dimension,lambda:rows),
                ConstraintRowBlock(len(phi),self.dimension,lambda:-rows)),
            groups=np.r_[geometry.groups,np.full(2*len(phi),group,dtype=np.int64)])
        self._base_geometry,self._surface_geometry=geometry,result
        return result


@dataclass(frozen=True)
class SurfaceAnchorMotion:
    motion: object
    extra_dimension: int

    @property
    def durations(self):
        return self.motion.durations

    def motion_terms(self, states):
        residual,rows = self.motion.motion_terms(states)
        return residual,np.pad(rows,((0,0),(0,self.extra_dimension)))


