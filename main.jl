include("nlp.jl")

corpus = ["Blue blue blue sky",
          "Look at the sky",
          "sky is blue",
          "The Eagle is in the sky",
          "Hello blue eagle",
          "Northern avenue"];


tfidf_ = tf_idf(corpus)
bow = bag_of_words(corpus)


print(tfidf_)

println()

println("Bag of words: ")
print(bow)
