import nltk
import embedding

def sliding_window(items, n = 3):
    pad = [None] * (n - 1)
    items = pad + items + pad
    for i in xrange(len(items) - n + 1):
    	yield items[i:i+n]


def main(corpus = "to be or not to be that is the question", m = 20):
	tokens = nltk.word_tokenize(corpus)
	vocabulary = embedding.Index(tokens)
	e = embedding.Embedding(vocabulary, m)
	print(vocabulary)
	print(e)
	for window in sliding_window(tokens):
		context = window[:-1]
		context_embedding = e[context]
		token = window[-1]
		token_embedding = vocabulary[[token]]
		print("%s\t%s" % (context, token))
		print("%s\t%s" % (context_embedding, token_embedding))


if __name__ == '__main__':
	main()