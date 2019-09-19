#Author: Apala Guha

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, Window, DataFrameReader, DataFrame, SQLContext
from pyspark.sql.types import StructField, StructType, StringType, IntegerType, ArrayType, DoubleType, FloatType, BooleanType
from pyspark.sql.functions import ltrim, shiftRightUnsigned, lead, udf, explode, split, repeat, concat, length, greatest, least, lit, concat_ws, approxCountDistinct, size, sum, create_map, collect_list, rand, floor, max, row_number, round, substring_index, spark_partition_id, monotonically_increasing_id, when, col, expr, array, broadcast, count, posexplode, when, collect_set, isnull, regexp_replace, trim
from pyspark.ml.feature import NGram, Word2Vec, Word2VecModel, MaxAbsScaler, VectorAssembler, Normalizer, VectorSlicer, MinMaxScaler, StandardScaler, QuantileDiscretizer, Bucketizer
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors, DenseVector
from pyspark.ml.param import TypeConverters
import sys
import __builtin__ as builtin

from dataset import *

def w2vFile (outDir, ctxSize, sampleSize, vecSize):
	return outDir + '/w2v_' + trainHistory + '_' + str(ctxSize) + '_' + str(sampleSize) + '_' + str(vecSize)

def lrmFile (outDir, ctxSize, sampleSize, vecSize, dim):
	return outDir + '/model_' + trainHistory + '_' + str(ctxSize) + '_' + str(sampleSize) + '_' + str(vecSize) + '_' + str(dim)

def outputFile(outDir, ctxSize, vecSize, sampleSizes):
	return outDir + '/accuracy_' + trainHistory + '_' + testHistory + '_' + str(ctxSize) + '_' + str(vecSize) + '_' + str(len(sampleSizes))

def ForceCompute(old, df, msg, flag=True, count=2):
	print '**********', msg
	df.cache()
	print df.rdd.getNumPartitions(), 'partitions'
	df.show(count, flag)

	if old != None:
		old.unpersist()



def CreateSubstring(df, inCol, outCol, strLen, delim, startPos, endPos, makeList=False):

	if endPos <= startPos:
		df = df.withColumn(outCol, lit(''))

	#here we create a substring of a string column
	startPos = builtin.min(builtin.max(0, startPos), strLen)
	endPos = builtin.min(builtin.max(startPos, endPos), strLen)

	#if one end of string coincides with beginning
	if startPos == 0:
		df = df.withColumn(outCol, substring_index(inCol, delim, endPos))
		

	#if  one end of string coincides with end
	elif endPos == strLen:
		df = df.withColumn(outCol, substring_index(inCol, delim, startPos - endPos))


	#if string is in middle
	else:
		#extract string from beginning upto position and then extract right end
		df = df.withColumn(outCol, substring_index(inCol, delim, endPos)) \
			.withColumn(outCol, substring_index(outCol, delim, startPos - endPos))

	#if string should be broken into list
	if makeList == True:
		df = df.withColumn(outCol, split(outCol, delim))

	return df





def BuildSubstringFeature(ngrams, w2v, start, end, ctxSize, lookahead, stringCol='sentence'):
	gramSize = GramSize(ctxSize, lookahead)

	vecass = VectorAssembler(outputCol='feature')

	old_ngrams = ngrams


	#create a vector column for each context position 
	for ctxpos in range(start, end):

		#create a column to hold the vector for this context position
		colName = 'ctx' + str(ctxpos)

		#create the vector for the context position
		ngrams = CreateSubstring(ngrams, stringCol, 'ngrams', gramSize, ' ', ctxpos, ctxpos + 1, True)
		

		ngrams = w2v.transform(ngrams).withColumnRenamed('vector', colName).drop('ngrams')
		
	
		if ctxpos == start:
			ngrams = vecass.setParams(inputCols = [colName]).transform(ngrams)
			ngrams = ngrams.withColumnRenamed('feature', 'tmp')
		else:
			ngrams = vecass.setParams(inputCols = ['tmp', colName]).transform(ngrams).drop('tmp')
			ngrams = ngrams.withColumnRenamed('feature', 'tmp')
			
		ngrams = ngrams.drop(colName)


	ngrams = ngrams.withColumnRenamed('tmp', 'feature')	
	return ngrams



def BuildFeatureVector(ngrams, w2v, ctxSize, lookahead):


	#transform ctx
	ngrams = BuildSubstringFeature(ngrams, w2v, 0, ctxSize, ctxSize, lookahead)



	return ngrams

def BuildLabelVector(ngrams, w2v, ctxSize, lookahead, nextPos):
	gramSize = GramSize(ctxSize, lookahead)

	#extract and transform label word
	ngrams = CreateSubstring(ngrams, 'sentence', 'ngrams', gramSize, ' ', ctxSize + nextPos, ctxSize + nextPos + 1, True) 
	ngrams = w2v.transform(ngrams).withColumnRenamed('vector', 'labelVec').drop('ngrams')


	return ngrams


def ClusterWords(w2v \
				, seqs
				):

	#to force each word to be a cluster center we use a trick
	#we train a kmeans model such that the number of clusters is equal to the number of words
	words = w2v.getVectors()

	


	words = words.join(broadcast(seqs), words.word == seqs.word).select(words.word.alias('word'), 'vector')
	words.cache()

	nwords = words.count()
	km = KMeans(featuresCol='vector', predictionCol='cluster', k=nwords)
	centers = km.fit(words)
	
	#create a dictionary of the words
	d = MakeDict(words, 'word', 'vector')

	old_words = words

	

	words = centers.transform(words) \
		.dropDuplicates(subset=['cluster']) \
		.withColumnRenamed('vector', 'centerVector')
		
	words.cache()
	words.show(10, False)
	


	return (words, d, centers)

def BuildPredictionVector(ngrams, lrmodels, ctxSize, vecSize):
	vecass = VectorAssembler(outputCol='vector')

	old_ngrams = ngrams

	#get prediction for each dimension
	incols = []
	for dim in range(0, vecSize):
		lrm = lrmodels[dim]

		colName = 'prediction' + str(dim)
		incols.append(colName)

		

		ngrams = lrm.transform(ngrams).withColumnRenamed('prediction', colName)

		if dim == 0:
			ngrams = vecass.setParams(inputCols = [colName]).transform(ngrams)
			ngrams = ngrams.withColumnRenamed('vector', 'tmp')
		else:
			ngrams = vecass.setParams(inputCols = ['tmp', colName]).transform(ngrams).drop('tmp')
			ngrams = ngrams.withColumnRenamed('vector', 'tmp')
			
		ngrams = ngrams.drop(colName)

		

	#end getting prediction for each dimension

	ngrams = ngrams.withColumnRenamed('tmp', 'vector')	
	ngrams = ngrams.drop('feature')




	return ngrams

def MakeDict(df, keycol, valcol):
	mymap = df.select(create_map(keycol, valcol).alias('map'))
	mylist = mymap.select(collect_list(mymap.map).alias('dict')).head()['dict']
	d = {}
	for elem in mylist:
		for key in elem:
			d[key] = elem[key]

	return d

def UpdateLatencies(ngrams):

	return ngrams


def Validate(ngrams \
			, sampleSizes \
			, ctxSize \
			, sqc \
			, seqs \
			, outFile \
			, minval \
			, maxval \
			, avg \
			, nlines):

	accuracy = []
	gramSize = GramSize(ctxSize, lookahead)

	c1 = (((maxval - minval) * 1.0) / nlines) / avg
	c2 = ((minval * 1.0) / nlines) / avg
	print seqs.count()
				


	ngrams = ngrams.repartition(1 << nPartLog)
	ngrams.cache()

	#we will validate separately for each vector size
	for vecSize in vecSizes:
		print '======TESTING FOR VECTOR SIZE', vecSize
		#start fresh
		old_ngrams = ngrams
		ngrams = ngrams.withColumn('correct', lit(0))



		#use models from each sample
		modelId = 0
		for sampleSize in sampleSizes:

			w2v = Word2VecModel.load(w2vFile(outDir, ctxSize, sampleSize, vecSize))
			lrmodels = []
			for dim in range(0, vecSize):
				lrmodels.append(LinearRegressionModel.load(lrmFile(outDir, ctxSize, sampleSize, vecSize, dim)))

			success = 0
			fail = 0
			unopt = 0

			#add columns to store model success and failure
			modelSucc = 'succ_' + str(modelId)
			modelFail = 'fail_' + str(modelId)
			modelUnopt = 'unopt_' + str(modelId)
			seqs = seqs.withColumn(modelSucc, lit(0)) \
						.withColumn(modelFail, lit(0)) \
						.withColumn(modelUnopt, lit(0))
			modelId = modelId + 1



			ngrams = ngrams \
				.withColumn('predSeq', lit(''))

			#create initial feature vector
			#transform each word into a cluster center
			words, d, centers = ClusterWords(w2v \
											, seqs \
											)
		
			#record correctness for this model only
			old_ngrams = ngrams
			ngrams = ngrams.withColumn('sample_correct', lit(0)).withColumn('sample_confi', lit(1.0))

			for nextPos in range(0,lookahead):
				#build the feature vector
				ngrams = BuildSubstringFeature(ngrams, w2v, nextPos, nextPos + ctxSize, ctxSize, lookahead,)

				#build the prediction vector
				ngrams = BuildPredictionVector(ngrams, lrmodels, ctxSize, vecSize)


			

				#now assign a cluster id to each prediction vector
				old_ngrams = ngrams
				ngrams = centers.transform(ngrams).withColumnRenamed('cluster', 'predWord').withColumnRenamed('vector', 'predictionVector')
				
				
				#get the predicted word
				ngrams = ngrams.join(broadcast(words), words.cluster == ngrams.predWord, 'inner') \
								.drop('cluster') #\

				#calculate the cosine similarity between prediction vector and center vector 
				epsilon = 0.0001
				def CosineSimi (v1, v2):
					d1 = DenseVector(v1)
					d2 = DenseVector(v2)
					n1 = d1.norm(2)
					n2 = d2.norm(2)
					return float(d1.dot(d2) / (n1 * n2))
				cossim = udf(lambda v1, v2: CosineSimi(v1, v2), DoubleType())
				ngrams = ngrams.withColumn('simi', cossim('centerVector', 'predictionVector'))
				ngrams = ngrams.drop('centerVector').drop('predictionVector')


				#update predicted sequence
				ngrams = ngrams.withColumn('predSeq', concat_ws(' ', 'predSeq', 'word')) 
				ngrams = ngrams.withColumn('predSeq', ltrim(ngrams.predSeq))


				#get actual sequence
				ngrams = CreateSubstring(ngrams, 'sentence', 'actualSeq', gramSize, ' ', ctxSize, ctxSize + nextPos + 1)


				#now get the cluster id for the predicted word in the sentence
				ngrams = BuildLabelVector(ngrams, w2v, ctxSize, lookahead, nextPos).withColumnRenamed('labelVec', 'vector').drop('ngrams')
				ngrams = centers.transform(ngrams).drop('vector')

				#and host latency for actual word
				ngrams = ngrams.join(broadcast(words), 'cluster', 'inner') \
						.drop('word') \
						.drop('centerVector') #\
				
				
			
				#record correctness
				ngrams = ngrams.withColumn('round_correct', when((ngrams.predWord != ngrams.cluster) | (ngrams.simi < confidence), 0).otherwise(nextPos + 1)).drop('predWord').drop('cluster')
				ngrams = ngrams.withColumn('sample_correct', when(ngrams.sample_correct + 1 == ngrams.round_correct, ngrams.round_correct).otherwise(ngrams.sample_correct)) 




				#get overall correctness
				ngrams = ngrams.withColumn('correct', greatest('sample_correct', 'correct'))

				#get binary correctness
				ngrams = ngrams.withColumn('binary_correct', when(ngrams.correct >= nextPos + 1, 1).otherwise(0))
				ngrams = ngrams.withColumn('sample_confi', when(ngrams.binary_correct == 1, 1.0).otherwise(least(ngrams.simi, ngrams.sample_confi)))
				ngrams = ngrams.withColumn('simi', when(ngrams.binary_correct == 1, ngrams.simi).otherwise(ngrams.sample_confi))


				ngrams = ngrams.withColumn('predSeq', when((ngrams.binary_correct == 1) | (ngrams.simi < confidence), ngrams.actualSeq).otherwise(ngrams.predSeq))
				ngrams = ngrams.withColumn('succ_wt', when(ngrams.binary_correct == 1, ngrams.wt).otherwise(0))
				ngrams = ngrams.withColumn('fail_wt', when((ngrams.binary_correct == 1) | (ngrams.simi < confidence), 0).otherwise(ngrams.wt))
				ngrams = ngrams.withColumn('unopt_wt', when((ngrams.binary_correct == 0) & (ngrams.simi < confidence), ngrams.wt).otherwise(0))
				ngrams = ngrams.drop('simi')

				#now summarize success and failure rates by predicted sequence
				seqWts = ngrams.groupBy('predSeq').agg(sum('succ_wt').alias('succ_wt'), sum('fail_wt').alias('fail_wt'), sum('unopt_wt').alias('unopt_wt'))

				#update sequences table
				seqs = seqWts.join(broadcast(seqs), seqWts.predSeq==seqs.word, 'right_outer').drop('predSeq').fillna(-c2/c1, ['succ_wt', 'fail_wt', 'unopt_wt'])


				scaleback = udf(lambda s: float(s*c1 + c2), DoubleType())
				seqs = seqs.withColumn(modelSucc, col(modelSucc) + scaleback(seqs.succ_wt)).drop('succ_wt')
				seqs = seqs.withColumn(modelFail, col(modelFail) + scaleback(seqs.fail_wt)).drop('fail_wt')
				seqs = seqs.withColumn(modelUnopt, col(modelUnopt) + scaleback(seqs.unopt_wt)).drop('unopt_wt')
				seqs.cache()

				aggregated = seqs.agg(sum(modelSucc), sum(modelFail), sum(modelUnopt))
				aggregated.cache()
				new_success = aggregated.head()['sum(' + modelSucc + ')']
				new_fail = aggregated.head()['sum(' + modelFail + ')']
				new_unopt = aggregated.head()['sum(' + modelUnopt + ')']
				print nextPos, new_success - success, new_fail - fail, new_unopt - unopt 
				success = new_success
				fail = new_fail
				unopt = new_unopt


		#end for testing for each model for a particular vector size

	#end for each vector size


	seqs.orderBy('succ_0', ascending=False).write.mode('overwrite').csv(outputFile(outDir, ctxSize, vecSize, sampleSizes))


	return accuracy

def main(f \
		, seqs, outDir
		):

	
	#read file
	df = ReadFile(f, spark, nPartLog)

	#preprocess file
	df = PreProcess(df, catSizeLog)

	sqc = SQLContext(sc, spark)

	df.cache()
	avg = df.agg({'freq':'avg'}).head()['avg(freq)']
	nlines = df.count()

	#train and test for each context size
	models = {}
	accuracy = [] 
	for ctxSize in ctxSizes:

		print '=============CTX SIZE', ctxSize, '================'

		#create dataset
		ngrams, minval, maxval = CreateDataset(df, ctxSize, lookahead, nPartLog, sc) #, nBuckets
		ngrams.cache()





		#test the models
		outFile = outDir + '/accuracy'
		accuracy.extend(Validate(ngrams, sampleSizes, ctxSize \
								, sqc \
								, seqs \
								, outFile \
								, minval \
								, maxval \
								, avg \
								, nlines))


		#ngrams.unpersist()
		sqc.clearCache()


	print accuracy
	

	return


conf = SparkConf().set('spark.dynamicAllocation.maxExecutors', '20')
sc = SparkContext(conf=conf)
spark = SparkSession(sc)
sc.setLogLevel("OFF")

f = sys.argv[1]
outDir = sys.argv[2]
seqFile = sys.argv[3]
nPartLog = int(sys.argv[4])
trainHistory = sys.argv[5]

lookahead = int(sys.argv[6])


#get ctxsizes
ctxSizes = []
argctr = 7
nargs = int(sys.argv[argctr])
for x in range(0, nargs):
	argctr = argctr + 1
	ctxSizes.append(int(sys.argv[argctr]))



#get vecsizes
vecSizes = []
argctr = argctr + 1
nargs = int(sys.argv[argctr])
for x in range(0, nargs):
	argctr = argctr + 1
	vecSizes.append(int(sys.argv[argctr]))


#get samplesizes
sampleSizes = []
argctr = argctr + 1
nargs = int(sys.argv[argctr])
for x in range(0, nargs):
	argctr = argctr + 1
	sampleSizes.append(int(sys.argv[argctr]))

argctr = argctr + 1
testHistory = sys.argv[argctr]
argctr = argctr + 1
confidence = float(sys.argv[argctr])

switch = 10
catSizeLog = 10
seqs = spark.read.csv(seqFile, sep=',', schema=StructType([StructField('word', StringType(), False)]))
seqs.show(10, False)

main(f, seqs, outDir)


sys.exit()


