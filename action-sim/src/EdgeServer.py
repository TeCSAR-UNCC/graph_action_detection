from src import Options
from src.PersonIDType import PersonIDType

import numpy as np
from numpy import linalg as la
from operator import attrgetter as ag

class ObjectHistory():
    def __init__(self, personID, life=-1):
        self.personID = personID
        self.life = life

class EdgeServer():
    def __init__(self, opts, edge_nodes, max_feats=1):
        self.currIdx = 0
        self.edge_nodes = edge_nodes
        # self.table_size = 1000
        self.table = [] #[ObjectHistory(PersonIDType()) for i in range(self.table_size)]
        self.l2_thr = 3
        framerate = min(opts.source_framerate, opts.framerate)
        self.max_tab_life = framerate * 60 * 120 # 120 minutes
        self.unlock_cooldown = framerate * 30 # 30 seconds
        self.max_feats = max_feats

    def getNodeFromId(self, id):
        for node in self.edge_nodes:
            if id == node.id:
                return node
        return None

    def process_node_data(self):
        no_cam_tab_idxs = []
        cam_tab_idxs = []
        all_cam_sendQs = []

        # Fetch table idxs for where the currCam is known and organize by currCam
        for node in self.edge_nodes:
            cam_tab_idxs.append([])
        for idx in range(len(self.table)):
            tab_entry_currCam = self.table[idx].personID.currCam
            if tab_entry_currCam == -1:
                no_cam_tab_idxs.append(idx)
            for node_idx in range(len(self.edge_nodes)):
                node = self.edge_nodes[node_idx]
                if tab_entry_currCam == node.id:
                    cam_tab_idxs[node_idx].append(idx)

        # Process updates on IDs held by each edge node
        for node_idx in range(len(self.edge_nodes)):
            node = self.edge_nodes[node_idx]
            # Update any previous entries from this camera
            node_tab_idxs = cam_tab_idxs[node_idx]
            for tab_idx in node_tab_idxs:
                for personID in node.sendQ:
                    if personID.label == self.table[tab_idx].personID.label:
                        self.table[tab_idx].personID.currCam = personID.currCam
                        # self.table[tab_idx].personID.feats = personID.feats
                        for i in range(personID.feats.shape[1]):
                            if self.table[tab_idx].personID.feats.shape[1] < self.max_feats:
                                det_feat_i = personID.feats[:,i].reshape((1280,1))
                                self.table[tab_idx].personID.feats = np.append(self.table[tab_idx].personID.feats,det_feat_i,axis=1)
                            else:
                                self.table[tab_idx].personID.feats[:,self.table[tab_idx].personID.featidx] = personID.feats[:,i]
                                self.table[tab_idx].personID.featidx = (self.table[tab_idx].personID.featidx+1) % self.max_feats
                        self.table[tab_idx].personID.bbox = personID.bbox
                        if personID.lock == 1:
                            self.table[tab_idx].life = self.max_tab_life
                            edge_update = [personID.label, self.table[tab_idx].personID.label, self.table[tab_idx].personID.feats, self.table[tab_idx].personID.featidx]
                            node.recvQ.append(edge_update)
                        elif self.unlock_cooldown == 0:
                            self.table[tab_idx].life = self.max_tab_life
                            self.table[tab_idx].personID.currCam = -1
                            no_cam_tab_idxs.append(tab_idx)
                        else:
                            self.table[tab_idx].life = self.unlock_cooldown
                            no_cam_tab_idxs.append(tab_idx)
                        node.sendQ.remove(personID)

            # Append remaining ids into all_cams queue
            for personID in self.edge_nodes[node_idx].sendQ:
                all_cam_sendQs.append(personID)

        # Search for ID matches among IDs with no current camera association
        num_cam_dets = len(all_cam_sendQs)
        num_no_cam_tabs = len(no_cam_tab_idxs)
        matched_det_idxs = []
        matched_tab_idxs = []
    
        if (num_cam_dets>0) and (num_no_cam_tabs>0):
            # Construct and fill match table
            match_table = np.full((num_cam_dets,num_no_cam_tabs), np.inf, dtype=float)
            for d in range(num_cam_dets):
                det_feat_idx = all_cam_sendQs[d].featidx
                det_feats = all_cam_sendQs[d].feats[:,det_feat_idx].reshape((1280,1))
                det_currCam = all_cam_sendQs[d].currCam
                for t in range(num_no_cam_tabs):
                    tab_idx = no_cam_tab_idxs[t]
                    tab_feats = self.table[tab_idx].personID.feats
                    tab_label = self.table[tab_idx].personID.label
                    tab_currCam = self.table[tab_idx].personID.currCam
                    if (tab_label != -1) and ((tab_currCam == -1) or (tab_currCam == det_currCam)):
                        match = np.mean(la.norm((tab_feats-det_feats), ord=2, axis=0))
                        if match < self.l2_thr:
                            match_table[d,t] = match

            best_match = match_table.min()
            while best_match < self.l2_thr:
                match_pos = np.where(match_table == best_match)

                # Fetch information for detection being entered into the table
                d = match_pos[0][0] # Index in valid_detections list
                det = all_cam_sendQs[d]

                # Fetch id entry in table
                t = match_pos[1][0] # Index in active_table_idxs list
                tab_idx = no_cam_tab_idxs[t]
                tab = self.table[tab_idx]

                self.table[tab_idx].personID.currCam = det.currCam
                # self.table[tab_idx].personID.feats = det.feats
                for i in range(det.feats.shape[1]):
                    if self.table[tab_idx].personID.feats.shape[1] < self.max_feats:
                        det_feat_i = det.feats[:,i].reshape((1280,1))
                        self.table[tab_idx].personID.feats = np.append(self.table[tab_idx].personID.feats,det_feat_i,axis=1)
                    else:
                        self.table[tab_idx].personID.feats[:,self.table[tab_idx].personID.featidx] = det.feats[:,i]
                        self.table[tab_idx].personID.featidx = (self.table[tab_idx].personID.featidx+1) % self.max_feats
                self.table[tab_idx].personID.bbox = det.bbox
                self.table[tab_idx].life = self.max_tab_life

                if det.lock == 1:
                    edge_update = [det.label, self.table[tab_idx].personID.label, self.table[tab_idx].personID.feats, self.table[tab_idx].personID.featidx]
                    node = self.getNodeFromId(det.currCam)
                    node.recvQ.append(edge_update)

                # Remove detection and table indices from match_table
                match_table[d,:] = np.full(match_table[d,:].shape, np.inf, dtype=float)
                match_table[:,t] = np.full(match_table[:,t].shape, np.inf, dtype=float)
                matched_det_idxs.append(d)
                matched_tab_idxs.append(t)

                # Calculate new best match
                best_match = match_table.min()

            # Clear matched detections from list of remaining valid detections
            for d in matched_det_idxs:
                all_cam_sendQs[d] = 'MATCHED'
            while 'MATCHED' in all_cam_sendQs:
                all_cam_sendQs.remove('MATCHED')

            # Clear matched table entries from list of table entries w/o a valid camera
            for t in matched_tab_idxs:
                no_cam_tab_idxs[t] = 'MATCHED'
            while 'MATCHED' in no_cam_tab_idxs:
                no_cam_tab_idxs.remove('MATCHED')

        # Add remaining detections from edge nodes to table
        while len(all_cam_sendQs) > 0:
            det = all_cam_sendQs.pop(0)
            if (len(no_cam_tab_idxs) > 0):
                tab_idx = no_cam_tab_idxs.pop(0)
                self.table[tab_idx].personID = det
                self.table[tab_idx].life = self.max_tab_life
            else:
                self.table.append(ObjectHistory(det, self.max_tab_life))
                tab_idx = -1
            #     tab_idx = self.table.index(min(self.table, key=ag('life')))
            # self.table[tab_idx].personID = det
            # self.table[tab_idx].life = self.max_tab_life
            
            if det.currCam != -1:
                edge_update = [det.label, det.label, self.table[tab_idx].personID.feats, self.table[tab_idx].personID.featidx]
                node = self.getNodeFromId(det.currCam)
                node.recvQ.append(edge_update)

        # # Update table entries that still have no valid camera
        # for tab_idx in no_cam_tab_idxs:
        #     if self.table[tab_idx].life > 0:
        #         self.table[tab_idx].life -= 1
        #     if self.table[tab_idx].life == 0:
        #         self.table[tab_idx] = ObjectHistory(PersonIDType())
        for tab in self.table:
            if tab.life > 0:
                tab.life -= 1
            if tab.life == 0:
                if tab.personID.lock == 0:
                    tab.personID.currCam = -1
                    tab.life = self.max_tab_life - self.unlock_cooldown
                else:
                    self.table.remove(tab)