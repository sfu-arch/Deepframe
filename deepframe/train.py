#Author: Apala Guha

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, Window, DataFrameReader, DataFrame, SQLContext
from pyspark.sql.types import StructField, StructType, StringType, IntegerType, ArrayType, DoubleType, FloatType, BooleanType
from pyspark.sql.functions import ltrim, shiftRightUnsigned, lead, udf, explode, split, repeat, concat, length, greatest, least, lit, concat_ws, approxCountDistinct, size, sum, create_map, collect_list, rand, floor, max, row_number, round, substring_index, spark_partition_id, monotonically_increasing_id, when, col, expr, array, broadcast, count, posexplode, when, collect_set, isnull
from pyspark.ml.feature import NGram, Word2Vec, Word2VecModel, MaxAbsScaler, VectorAssembler, Normalizer, VectorSlicer, MinMaxScaler, StandardScaler, QuantileDiscretizer, Bucketizer
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors, DenseVector
from pyspark.ml.param import TypeConverters
import sys
import __builtin__ as builtin

import dataset
from dataset import *

def w2vFile (outDir, ctxSize, sampleSize, vecSize):
	return outDir + '/w2v_' + trainHistory + '_' + str(ctxSize) + '_' + str(sampleSize) + '_' + str(vecSize)

def lrmFile (outDir, ctxSize, sampleSize, vecSize, dim):
	return outDir + '/model_' + trainHistory + '_' + str(ctxSize) + '_' + str(sampleSize) + '_' + str(vecSize) + '_' + str(dim)

def outputFile(outDir, ctxSize, vecSize, sampleSizes):
	return outDir + '/accuracy_' + testHistory + '_' + str(ctxSize) + '_' + str(vecSize) + '_' + str(len(sampleSizes))

def ForceCompute(old, df, msg, flag=True, count=2):
	print '**********', msg
	df.cache()
	print df.rdd.getNumPartitions(), 'partitions'
	df.show(count, flag)

	if old != None:
		old.unpersist()



def CreateSubstring(df, inCol, outCol, strLen, delim, startPos, endPos, makeList=False):
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



def FormSample(ngrams, cnt, sId \
				):

	ngrams_thresh = sc.parallelize(ngrams.orderBy('wt', ascending=False).head(cnt * (sId + 1))).toDF()#.groupBy('sentence').sum('wt').withColumnRenamed('sum(wt)', 'wt')
	return ngrams_thresh

	#create a stratified sample based on the number of buckets
	#calculate the overall sampling fraction
	total = ngrams.count()
	cnt = min(cnt, total)
	sampFrac = builtin.min(1.0, cnt * nBuckets * 1.0 / total)
	print cnt, total, nBuckets, sampFrac

	#now get the fraction for each strata
	#add up their relative fractions
	#highest quantile gets more representation
	relative = 1.0
	relSum = 0.0
	for i in range(0, nBuckets):
		relative = 1.0 / (nBuckets - i)
		relSum += relative

	#now calculate the base fraction
	baseFrac = sampFrac / relSum


	#create the strata
	d = {}
	for i in range(0, nBuckets):
		d[i] = baseFrac / (nBuckets - i)
		print i, d[i]

	#sample by strata
	ngrams_thresh = ngrams.sampleBy('bucket', d, seed=42).drop('bucket')

	return ngrams_thresh


	
def BuildWordModel(ngrams, ctxSize, vecSize, lookahead, nextPos):

	#halve the context size because of the way word2vec behaves
	windowSize = ctxSize / 2
	gramSize = GramSize(ctxSize, lookahead)
	

	#create relevant substrings
	ngrams = CreateSubstring(ngrams, 'sentence', 'ctxLeft', gramSize, ' ', 0, windowSize)
	ngrams = CreateSubstring(ngrams, 'sentence', 'ctxRight', gramSize, ' ', windowSize, windowSize + windowSize)
	ngrams = CreateSubstring(ngrams, 'sentence', 'word', gramSize, ' ', windowSize + windowSize + nextPos, windowSize + windowSize + nextPos + 1)
	ngrams = ngrams.withColumn('ngrams', concat_ws(' ', 'ctxLeft', 'word')).withColumn('ngrams', concat_ws(' ', 'ngrams', 'ctxRight'))
	ngrams = ngrams.withColumn('ngrams', split('ngrams', ' '))

	#train model
	word2vec = Word2Vec(windowSize=ctxSize, vectorSize=vecSize, seed=42, inputCol='ngrams', outputCol='vector')
	model = word2vec.fit(ngrams)

	return model

def BuildSubstringFeature(ngrams, w2v, start, end, ctxSize, lookahead):
	gramSize = GramSize(ctxSize, lookahead)

	vecass = VectorAssembler(outputCol='feature')

	old_ngrams = ngrams


	#create a vector column for each context position 
	for ctxpos in range(start, end):

		#create a column to hold the vector for this context position
		colName = 'ctx' + str(ctxpos)

		#create the vector for the context position
		ngrams = CreateSubstring(ngrams, 'sentence', 'ngrams', gramSize, ' ', ctxpos, ctxpos + 1, True)
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

def BuildDimModel(ngrams, dim):
	#get label value for this dimension
	createlabel = udf(lambda s: float(s[dim]), DoubleType())
	ngrams_train = ngrams.withColumn('label', createlabel('labelVec'))

	#build LR model
	lr = LinearRegression(featuresCol='feature', labelCol='label', predictionCol='prediction', weightCol='wt', solver='l-bfgs')
	lrm = lr.fit(ngrams_train)
	print lrm.coefficients, lrm.intercept

	return lrm

		

def BuildLRModels(ngrams, w2v, ctxSize, vecSize, lookahead, nextPos):
	#build the label vector
	old_ngrams = ngrams
	ngrams = BuildLabelVector(ngrams, w2v, ctxSize, lookahead, nextPos)


	#build the feature vector	
	ngrams = BuildFeatureVector(ngrams, w2v, ctxSize, lookahead)


	#now build one LR model for each dimension
	lrmodels = []
	for dim in range(0, vecSize):
		lrm = BuildDimModel(ngrams, dim)
		lrmodels.append(lrm)

	#end computing LR models for each dim


	return lrmodels

def Train(ngrams, ctxSize \
			, outDir
			):
	#train on different sample sizes taken from the top of the ngrams table
	models = {}
	for sampleSize in sampleSizes:

		print '======TRAINING FOR SAMPLE SIZE', sampleSize

		#form sample
		ngrams_thresh = FormSample(ngrams, sampleSize, 0 \
									#, nBuckets
									)
		ngrams_thresh.cache()

	
		#form models for each vector size on the sample
		for vecSize in vecSizes:
			#even if we lookahead multiple paths
			#only create models for one lookahead
			#to make use of maximum context
			nextPos = 0
			
			print '======TRAINING FOR VECTOR SIZE', vecSize

			#learn word2vec model
			w2v = BuildWordModel(ngrams_thresh, ctxSize, vecSize, lookahead, nextPos)

			#learn regression models using the above dictionary
			lrmodels = BuildLRModels(ngrams_thresh, w2v, ctxSize, vecSize, lookahead, nextPos)


			outFile = w2vFile(outDir, ctxSize, sampleSize, vecSize)
			w2v.write().overwrite().save(outFile)
			for dim in range(0, vecSize):
				outFile = lrmFile(outDir, ctxSize, sampleSize, vecSize, dim)
				lrmodels[dim].write().overwrite().save(outFile)


			#remember models for this sample and vector size
			try: #for subsequent samples
				models[vecSize].append((w2v, lrmodels))
			except: #for first sample
				models[vecSize] = [(w2v, lrmodels)]

		#end training for each vector size on a given sample

	#end training on different sample sizes		

	return models



def main(f \
		, outDir
		):

	
	#read file
	df = ReadFile(f, spark, nPartLog)
	nlines = df.count()
	
	#preprocess file
	df = PreProcess(df, catSizeLog)

	sqc = SQLContext(sc, spark)

	df.cache()

	#train and test for each context size
	models = {}
	accuracy = [] 
	for ctxSize in ctxSizes:

		print '=============CTX SIZE', ctxSize, '================'

		#create dataset
		ngrams = CreateDataset(df, ctxSize, lookahead, nPartLog, sc) [0]#, nBuckets
		ngrams.cache()

		#build models for a particular context size
		vecModels = Train(ngrams, ctxSize, outDir)
						#, nBuckets

		models[ctxSize] = vecModels






		sqc.clearCache()
	#end train and test for each context size


	print accuracy
	

	return


conf = SparkConf().set('spark.dynamicAllocation.maxExecutors', '20')
sc = SparkContext(conf=conf)
spark = SparkSession(sc)
sc.setLogLevel("OFF")

f = sys.argv[1]
outDir = sys.argv[2]
nPartLog = int(sys.argv[3])

trainHistory = sys.argv[4]
lookahead = int(sys.argv[5])

#get ctxsizes
ctxSizes = []
argctr = 6
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




switch = 10
catSizeLog = 10

main(f, outDir)


sys.exit()


