using LinearAlgebra
using DataFrames

# ------------------------------------------------------------------------------------------------------------------

function tf_idf(corpus)
    
    
    N_corpus = length(corpus)

    bw = [[lowercase(split(corpus[j])[i]) for i in 1:length((split(corpus[j])))] for j in 1:N_corpus]
    
    bw_union = union(vcat(bw...))
    
    # ===========================================================================================

    
    C_vec = []

    for i in 1:N_corpus

        c = Dict(zip(bw_union, zeros(length(bw_union))))

        for word in bw_union

            if word in bw[i]

                idx = findall(x->x==word, bw[i])
                nidx = length(findall(x->x==word, bw[i]))
                c[word] += nidx
                deleteat!(bw[i], findall(x->x==word, idx))

            else

                c[word] = 0
            end

        end

        push!(C_vec, c)

    end
    
    
    
    # ===========================================================================================

    
    idf = Dict(x => log((1 + N_corpus)/(1 + sum([(x in bw[i]) for i in 1:N_corpus ]))) + 1 for x in bw_union)
    
    tfidf = [values(C_vec[i]) .* values(idf)/norm(values(C_vec[i]) .* values(idf)) for i in 1:N_corpus]
    
    cols = [key for key in keys(C_vec[1])]
    
    data_dic = Dict(cols[i] => hcat(tfidf...)[i, 1:N_corpus] for i in 1:length(cols))
    
    df = DataFrame(data_dic)

    return df
    
end


function bag_of_words(corpus)
    
    _word_count = Dict()
    
    N_corpus = length(corpus)
    
    bw = [[lowercase(split(corpus[j])[i]) for i in 1:length((split(corpus[j])))] for j in 1:N_corpus]
    
    bw = vcat(bw...)
    
    for word in bw
    
        if haskey(_word_count, word)
        
            _word_count[word] += 1
        
        else
        
            _word_count[word] = 1
        
        end
        
    end
    
    return _word_count
    
end

