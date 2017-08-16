import numpy
import scipy

def calc_NN_estimator(gen_scores, imp_scores):
	# The Nearest Neighbour estimator of the KL-divergence, to calculate the relative entropy based on the input genuine and impostor match scores

	epsilon = 10 ** (-10) # small number to add to one of the match scores if the compared match scores are the same (to avoid division by 0 in log equation)

	# Calculate distance between each genuine score and its nearest neighbour amonst the genuine scores
	print('Calculating genuine-genuine NN distances')
	#print(gen_scores)
	gen_NN_dists = numpy.zeros(len(gen_scores))
	min_gen_dist = float('inf')
	for gs_i in range(len(gen_scores)):
		for gs_j in range(len(gen_scores)):
			if (gs_i != gs_j):
				crnt_gen_dist = abs(gen_scores[gs_i] - gen_scores[gs_j])
				if crnt_gen_dist < min_gen_dist:
					min_gen_dist = crnt_gen_dist
					if min_gen_dist == 0:
						print('0 distance between %s (%s) and %s (%s)' % (gs_i, gen_scores[gs_i], gs_j, gen_scores[gs_j]))
						min_gen_dist = epsilon
				elif crnt_gen_dist >= min_gen_dist:
					continue
			elif (gs_i == gs_j):
				continue
		gen_NN_dists[gs_i] = min_gen_dist
		min_gen_dist = float('inf')
	#print(gen_NN_dists)

	# Calculate distance between each genuine score and its nearest neighbour amongst the impostor scores
	print('Calculating genuine-impostor NN distances')
	imp_NN_dists = numpy.zeros(len(gen_scores))
	min_imp_dist = float('inf')
	for gs_i in range(len(gen_scores)):
		for is_j in range(len(imp_scores)):
			crnt_imp_dist = abs(gen_scores[gs_i] - imp_scores[is_j])
			if crnt_imp_dist < min_imp_dist:
				min_imp_dist = crnt_imp_dist
				if min_imp_dist == 0:
					print('0 distance between %s (%s) and %s (%s)' % (gs_i, gen_scores[gs_i], is_j, imp_scores[is_j]))
					min_imp_dist = epsilon
			elif crnt_imp_dist >= min_imp_dist:
				continue
		imp_NN_dists[gs_i] = min_imp_dist
		min_imp_dist = float('inf')
	#print(imp_NN_dists)

	# Calculate the KL-divergence estimation (relative entropy)
	rel_entr = (1 / float(len(gen_scores))) * sum(scipy.log2(imp_NN_dists / gen_NN_dists)) + scipy.log2(len(imp_scores) / float((len(gen_scores) - 1)))

	return rel_entr
