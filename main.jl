include("nlp.jl")

corpus = ["Blue blue blue sky",
          "Look at the sky",
          "sky is blue",
          "The Eagle is in the sky",
          "Hello blue eagle",
          "Northern avenue"];


tfidf_ = tf_idf(corpus)
bow = bag_of_words(corpus)
cosine_matrix = cosine_similarity_matrix(corpus)


println("TF-TDF: ")
print(tfidf_)

println()

println("Bag of words: ")
println(bow)

println("Cosine of similarity matrix: ")
println(cosine_matrix)
