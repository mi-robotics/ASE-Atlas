import numpy as np
import torch
import faiss 
import time
#TODO -> check how the amp ordering works
#TODO -> ensure min steps is 10 given latents


'''
Core Idea:

Store all latents and transitions

'''

class TripletReward:

    def __init__(self):
        return
    

    def _get_contrastive_samples(self, ase_latents, amp_obs):
        '''
        latents -> 32, num_envs, latent dim
        amp_obs -> 32, num_envs, amb_obs_dim (10,109)
        '''
        # ------------------------------------------------------------  data formatting

        ase_latents = ase_latents.transpose(1,0).reshape(
            ase_latents.size(0)*ase_latents.size(1), -1
        ) # -->  num_envs*32, latent_dims

        amp_obs = amp_obs.transpose(1,0).reshape(
            amp_obs.size(0)*amp_obs.size(1), amp_obs.size(2), -1
        ) # --> num_envs*32, 10, 109

        
        # ------------------------------------------------------------ latent mapping 
        # we need some mapping between latents and amp obs
        latents_diffs = torch.diff(ase_latents, dim=0) # col wise changes
        latents_diffs = torch.any(latents_diffs !=0, dim=1) #check for changes on a row
        changes_idxs = torch.where(latents_diffs)[0] + 1

        repeats = torch.cat([
            # Count for the first segment
            torch.Tensor([changes_idxs[0]]),
            # Counts for intermediate segments
            torch.diff(changes_idxs),
            # Count for the last segment
            torch.Tensor([ase_latents.size(0) - changes_idxs[-1]])
        ])

        print(len(changes_idxs))
        print(len(repeats))
        input()

        #groups latents by the latents
        latent_amp_obs = []
        start_idx = 0

        for idx in changes_idxs:
            latent_amp_obs.append(amp_obs[start_idx:idx])
            start_idx = idx
        latent_amp_obs.append(amp_obs[start_idx:]) # --> [num latents, latent_seq_len, amp_obs.size()]

        #pair the latents
        unique_latents = []
        start_idx = 0
      
        for idx in changes_idxs:
            unique_latents.append(ase_latents[idx-1])
            start_idx = idx
        unique_latents.append(ase_latents[-1])
        
        unique_latents = torch.stack(unique_latents)

        # Reconstruct the original sequence
        reconstructed_sequence = torch.repeat_interleave(unique_latents, repeats.int(), dim=0)

        print(torch.all(reconstructed_sequence == ase_latents))
        input()


        assert len(latent_amp_obs) == unique_latents.size(0)

        # ------------------------------------------------------------ create dataset 
        # Convert the PyTorch tensor to a NumPy array for FAISS
        dataset_latents = unique_latents.numpy()

        start = time.time()

        # Create a basic L2 index
        base_index = faiss.IndexFlatL2(24)

        # Wrap the base index with IndexIDMap2
        index = faiss.IndexIDMap2(base_index)

        # Assign an ID to each vector. In this case, use a range of IDs.
        ids = np.arange(unique_latents.size(0), dtype=np.int64)

        # Add vectors and their IDs to the index
        index.add_with_ids(dataset_latents, ids)

        # ------------------------------------------------------------ query dataset 
        k = 2
        D, I = index.search(dataset_latents, k) # --> return a nearest n for each time steps
        
        nearest_n = I[:, 1] # removes the exact match

        nearest_n = torch.repeat_interleave(torch.from_numpy(nearest_n), repeats.int(), dim=0)


        end = time.time()

        print('executiuon time:', end-start)
        input()


        # ------------------------------------------------------------ colelct possitive and negative samples 
        positive_batch_amp_obs = [latent_amp_obs[ni] for ni in nearest_n]
        negative_batch_amp_obs = [latent_amp_obs[ri] for ri in np.random.randint(0, unique_latents.size(0), ase_latents.size(0))]


        positive_amp_obs = []
        negative_amp_obs = []

        for i in range(len(positive_batch_amp_obs)):
            pos_samples = positive_batch_amp_obs[i].size(0)
            neg_samples = negative_batch_amp_obs[i].size(0)

            pos_obs = positive_batch_amp_obs[i][np.random.randint(0, pos_samples)]            
            neg_obs = negative_batch_amp_obs[i][np.random.randint(0, neg_samples)]

            positive_amp_obs.append(pos_obs)
            negative_amp_obs.append(neg_obs)


        anchor = amp_obs
        positive_amp_obs = torch.cat(positive_amp_obs)
        negative_amp_obs = torch.cat(negative_amp_obs)

        model_inputs = torch.cat((anchor, positive_amp_obs, negative_amp_obs))

        


    

if __name__ == '__main__':

    
    latents = torch.randn((1, 10000, 24))
    latents = latents.repeat((32,1,1))
    amp_obs = torch.randn((32, 6000, 10, 109))

    trip = TripletReward()

    trip._get_contrastive_samples(latents, amp_obs)


