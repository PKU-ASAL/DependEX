from wordsegment import load, segment
import similarity_score
import sys
import gensim.downloader

def split_words():
    load()
    word_list = segment('firstname')
    print(word_list)

def similar_semantics():
    phrase_list_1 = ["user's own name"]
    phrase_list_2 = ["city", "last name", "phone", "full name", "password", "country", "user", "email", "state","first name", "age"]
    model = similarity_score.load_model()
    for phrase1 in phrase_list_1:
        max_score = -1
        max_phrase = 'null'
        for phrase2 in phrase_list_2:
            score = similarity_score.score(phrase1, phrase2, model)
            if score>max_score:
                max_score = score
                max_phrase = phrase2
        print(phrase1,' ',max_phrase,' max score: ',max_score)
    print('finished')

if __name__ == "__main__":
    similar_semantics()