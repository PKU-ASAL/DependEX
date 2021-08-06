from nltk.tokenize import word_tokenize
import gensim

def load_model():
	model = gensim.models.KeyedVectors.load_word2vec_format('/home/zhangsk/gensim-data/word2vec-google-news-300/GoogleNews-vectors-negative300.bin', binary=True)
	return model

def score(seq1, seq2, model,tail = 100, head = 10):
	seq1_word_list = word_tokenize(seq1.strip().lower())[-tail:]
	seq2_word_list = word_tokenize(seq2.strip().lower())[:head]
	return sim_score(seq1_word_list, seq2_word_list,model)

def sim_score(wordlist1, wordlist2,model):
	maxes = []
	for word in wordlist1:
		cur_max = 0
		for word2 in wordlist2:
			if word == word2:
				sim = 1
				cur_max = sim
			elif word in model.vocab and word2 in model.vocab:
				sim = model.similarity(word, word2)
				if sim > cur_max:
					cur_max = sim
		if cur_max != 0:
			maxes.append(cur_max)
	if sum(maxes) == 0:
		return 0
	return float(sum(maxes)) / len(maxes)