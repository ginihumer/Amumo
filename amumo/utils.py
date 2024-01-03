from .model import get_model
from . import model as am_model
import scipy.cluster.hierarchy as sch
from sklearn.feature_extraction.text import CountVectorizer
import os
import numpy as np
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

# cach data for faster interaction
data_checkpoint_dir = 'data_checkpoints/'

if not os.path.exists(data_checkpoint_dir):
    os.makedirs(data_checkpoint_dir)

def get_embeddings_per_modality(model, dataset_name, all_data, batch_size = 500):
    batch_size = min(len(all_data[list(all_data.keys())[0]]), batch_size)

    if type(model) == str:
        clip_model = get_model(model, device=device)
    elif issubclass(type(model), am_model.CLIPModelInterface):
        clip_model = model
    else:
        print('model_name must either be a string or of type CLIPModelInterface')

    data_prefix = dataset_name + '_' + clip_model.model_name + '_' + clip_model.name
    data_prefix = data_prefix.replace('/','-')

    all_features = {}

    for modality in all_data.keys():
        if not os.path.exists(data_checkpoint_dir + data_prefix + '_' + modality + '-embedding.csv'):
            all_data_of_modality = all_data[modality]
            with torch.no_grad():
                batch_features = []
                for i in range(int(len(all_data_of_modality)/batch_size)):
                    print("batch", i+1, "of", int(len(all_data_of_modality)/batch_size))
                    batch = all_data_of_modality[i*batch_size:(i+1)*batch_size]
                    # check if model is able to encode this modality
                    if clip_model.encoding_functions[modality] is None:
                        print('no encoding function for', modality)
                        break
                    batch_features.append(clip_model.encoding_functions[modality](batch).float().cpu())
                    
                features = torch.cat(batch_features, dim=0)

            np.savetxt(data_checkpoint_dir + data_prefix + '_' + modality + '-embedding.csv', features.cpu(), delimiter = ',')
        else:
            print('found cached embeddings for', data_prefix, modality)
            features = torch.from_numpy(np.genfromtxt(data_checkpoint_dir + data_prefix + '_' + modality + '-embedding.csv', delimiter=","))

        all_features[modality] = features/features.norm(dim=-1, keepdim=True)

    return all_features, clip_model.logit_scale

# deprecated; use "get_embeddings_per_modality" instead
def get_embedding(model_name, dataset_name, all_images, all_prompts, batch_size = 500):
    batch_size = min(len(all_images), batch_size)

    if type(model_name) == str:
        clip_model = get_model(model_name, device=device)
    elif issubclass(type(model_name), am_model.CLIPModelInterface):
        clip_model = model_name
    else:
        print('model_name must either be a string or of type CLIPModelInterface')

    data_prefix = dataset_name + '_' + clip_model.model_name + '_' + clip_model.name
    data_prefix = data_prefix.replace('/','-')
    if not os.path.exists(data_checkpoint_dir + data_prefix + '_image-embedding.csv') or not os.path.exists(data_checkpoint_dir + data_prefix + '_text-embedding.csv'):
        # with torch.no_grad():
        #     image_features = clip_model.encode_image(all_images).float()
        #     text_features = clip_model.encode_text(all_prompts).float()

        with torch.no_grad():
            batch_image_features = []
            batch_text_features = []
            
            for i in range(int(len(all_images)/batch_size)):
                print("batch", i+1, "of", int(len(all_images)/batch_size))
                batch_images = all_images[i*batch_size:(i+1)*batch_size]
                batch_prompts = all_prompts[i*batch_size:(i+1)*batch_size]

                batch_image_features.append(clip_model.encode_image(batch_images).float().cpu())
                batch_text_features.append(clip_model.encode_text(batch_prompts).float().cpu())
                
            image_features = torch.cat(batch_image_features, dim=0)
            text_features = torch.cat(batch_text_features, dim=0)

        np.savetxt(data_checkpoint_dir + data_prefix + '_image-embedding.csv', image_features.cpu(), delimiter = ',')
        np.savetxt(data_checkpoint_dir + data_prefix + '_text-embedding.csv', text_features.cpu(), delimiter = ',')
    else:
        print('found cached embeddings for', data_prefix)
        image_features = torch.from_numpy(np.genfromtxt(data_checkpoint_dir + data_prefix + '_image-embedding.csv', delimiter=","))
        text_features = torch.from_numpy(np.genfromtxt(data_checkpoint_dir + data_prefix + '_text-embedding.csv', delimiter=","))

    return image_features/image_features.norm(dim=-1, keepdim=True), text_features/text_features.norm(dim=-1, keepdim=True), clip_model.logit_scale


def get_similarity(image_features_norm, text_features_norm):
    # similarity between images and texts
    similarity = text_features_norm.cpu().numpy() @ image_features_norm.cpu().numpy().T # text x image 

    # similarity between all features
    features_norm = torch.cat((image_features_norm, text_features_norm), dim = 0)
    similarity_features = features_norm.cpu().numpy() @ features_norm.cpu().numpy().T
    
    return similarity, similarity_features



def get_cluster_sorting(similarity):
    # adapted from https://wil.yegelwel.com/cluster-correlation-matrix/
    linkage = sch.linkage(1-similarity, method='complete')
    # cluster_distance_threshold = (1-similarity_texts_images).max()/2 # np.percentile(1-similarity, 75)
    # idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, criterion='distance')
    idx_to_cluster_array = sch.fcluster(linkage, 10, criterion='maxclust')
    idx = np.argsort(idx_to_cluster_array)
    return idx, idx_to_cluster_array[idx], idx_to_cluster_array


def aggregate_texts(emb_ids, all_prompts, input_type=None):
    c_vec = CountVectorizer(ngram_range=(1, 77), stop_words="english")
    ngrams = c_vec.fit_transform([all_prompts[int(i)] for i in emb_ids])
    vocab = c_vec.vocabulary_
    count_values = ngrams.toarray().sum(axis=0)
    
    ngrams_sorted = sorted([(count_values[i],k) for k,i in vocab.items()], reverse=False, key=lambda sl: (len(sl[1]), sl[0]))
    
    ngrams_distinct = []
    for i in range(len(ngrams_sorted)):
        ng_count_a = ngrams_sorted[i][0]
        ng_text_a = ngrams_sorted[i][1]

        is_included = False
        for j in range(i+1, len(ngrams_sorted)):
            ng_count_b = ngrams_sorted[j][0]
            ng_text_b = ngrams_sorted[j][1]

            if ng_text_a in ng_text_b and ng_count_a <= ng_count_b:
                is_included = True
                break

        if not is_included:
            ngrams_distinct.append({"text": ng_text_a, "value": int(ng_count_a)})
    
    return (sorted(ngrams_distinct, key=lambda i: i['value'], reverse=True), "text-value")
    # return (sorted([{"value": int(count_values[i]), "text": k} for k,i in vocab.items()], key=lambda i: i["value"], reverse=True), "text-value")
    
def get_textual_label_for_cluster(emb_ids, all_prompts, k=2):
    ngrams,_ = aggregate_texts(emb_ids, all_prompts)
    return ' | '.join([ngram["text"] for ngram in ngrams[:k]])




###### Modality gap as defined in https://openreview.net/pdf?id=S7Evzt9uit3 ######

def l2_norm(x):
    return x/np.linalg.norm(x, axis=-1, keepdims=True)

def get_modality_gap(modal1_features, modal2_features):
    return modal1_features.mean(axis=0) - modal2_features.mean(axis=0)

def get_modality_gap_normed(modal1_features, modal2_features):
    # with the normed vector, we can use a delta value that defines how much we want to go in each direction
    return l2_norm(get_modality_gap(modal1_features, modal2_features))

def get_modality_distance(modal1_features, modal2_features):
    # Euclidean distance between mass centers
    return np.linalg.norm(get_modality_gap(modal1_features, modal2_features))

# This rotation approach is not legit... it uses the information that rows in the two modalities belong together
# import scipy.spatial as spa
# def get_modality_distance_rotated(modal1_features, modal2_features):
#     mtx1, mtx2, disparity = spa.procrustes(modal1_features, modal2_features)
#     return disparity

def calculate_val_loss(image_features_np, text_features_np, logit_scale = 100.0):
# give two lists of features, calculate loss

    # normalized features
    image_features_np /= np.linalg.norm(image_features_np, axis=-1, keepdims=True) + 1e-12
    text_features_np /= np.linalg.norm(text_features_np, axis=-1, keepdims=True) + 1e-12

    total_loss_list = list()

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    BATCH_SIZE = 50 # 5000 in total. 
    for idx in range(len(image_features_np)//BATCH_SIZE):
        
        with torch.no_grad():
            image_features = image_features_np[idx*50:(idx+1)*50]
            text_features  = text_features_np[idx*50:(idx+1)*50]      
        
            # cosine similarity as logits
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logits_per_image.T

            # # symmetric loss function
            labels = torch.arange(BATCH_SIZE,dtype=torch.long)#.cuda()
            loss_i = loss_img(logits_per_image, labels)
            loss_t = loss_txt(logits_per_text.T, labels)
            total_loss = (loss_i + loss_t)/2
            
            total_loss_list.append(total_loss.item())
    avg_val_loss = np.mean(total_loss_list)
    # print('avg_val_loss', avg_val_loss)
    return avg_val_loss

def calculate_cal_acc(modal1_features, modal2_features):
    return ((modal1_features @ modal2_features.T).argmax(axis=-1) == np.arange(len(modal2_features))).mean()


def get_closed_modality_gap(modal1_features, modal2_features):
    gap = get_modality_gap(modal1_features, modal2_features)

    modal1_modified = modal1_features - 0.5 * gap
    modal2_modified = modal2_features + 0.5 * gap

    # return modal1_modified, modal2_modified
    # in modality gap paper they norm the embeddings again, but this introduces a modality gap again...
    # TODO: check whether or not to norm
    return l2_norm(modal1_modified), l2_norm(modal2_modified)

# This rotation approach is not legit... it uses the information that rows in the two modalities belong together
# import scipy.linalg as alg
# def get_closed_modality_gap_rotated(modal1_features, modal2_features):
#     R, sca = alg.orthogonal_procrustes(modal1_features, modal2_features)
#     modal2_features_rot = np.dot(modal2_features, R.T) #* sca
#     return modal1_features, torch.from_numpy(modal2_features_rot)

def get_gap_direction(image_features_np, text_features_np, pca):
    image_features_np /= np.linalg.norm(image_features_np, axis=-1, keepdims=True) + 1e-12
    text_features_np /= np.linalg.norm(text_features_np, axis=-1, keepdims=True) + 1e-12

    pca_result = pca.transform(image_features_np)
    pca_result2 = pca.transform(text_features_np)
    pca_one_delta = pca_result2[:,0].mean() - pca_result[:,0].mean()

    return np.sign(pca_one_delta)