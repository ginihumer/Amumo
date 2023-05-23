from .model import get_model
import scipy.cluster.hierarchy as sch
from sklearn.feature_extraction.text import CountVectorizer
import os
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# cach data for faster interaction
data_checkpoint_dir = 'data_checkpoints/'

if not os.path.exists(data_checkpoint_dir):
    os.makedirs(data_checkpoint_dir)


def get_embedding(model_name, dataset_name, all_images, all_prompts):
    clip_model = get_model(model_name, device=device)

    data_prefix = dataset_name + '_' + clip_model.model_name + '_' + clip_model.name
    data_prefix = data_prefix.replace('/','-')
    if not os.path.exists(data_checkpoint_dir + data_prefix + '_image-embedding.csv') or not os.path.exists(data_checkpoint_dir + data_prefix + '_text-embedding.csv'):
        with torch.no_grad():
            image_features = clip_model.encode_image(all_images).float()
            text_features = clip_model.encode_text(all_prompts).float()

        np.savetxt(data_checkpoint_dir + data_prefix + '_image-embedding.csv', image_features.cpu(), delimiter = ',')
        np.savetxt(data_checkpoint_dir + data_prefix + '_text-embedding.csv', text_features.cpu(), delimiter = ',')
    else:
        print('found cached embeddings for', data_prefix)
        image_features = torch.from_numpy(np.genfromtxt(data_checkpoint_dir + data_prefix + '_image-embedding.csv', delimiter=","))
        text_features = torch.from_numpy(np.genfromtxt(data_checkpoint_dir + data_prefix + '_text-embedding.csv', delimiter=","))

    return image_features/image_features.norm(dim=-1, keepdim=True), text_features/text_features.norm(dim=-1, keepdim=True)


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
    cluster_distance_threshold = (1-similarity).max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, criterion='distance')
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