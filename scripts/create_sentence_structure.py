from SIF_model import *
from vocab_utils import load_vocab

sentence_structure_file = "../models/sentence_structure.npz"
args = parse_args()
vocab = load_vocab('../vocab_embedding/vocab_quora_train.txt')
sif_model.sent_indices, sif_model.sent_mask = sif_model.createStructure()
np.savez_compressed(sentence_structure_file, sent_indices = sif_model.sent_indices, sent_mask = sif_model.sent_mask)
