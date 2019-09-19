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


def outputFile(outDir, ctxSize):
	return outDir + '/sequence_' + trainHistory + '_' + str(ctxSize)

def ForceCompute(old, df, msg, flag=True, count=2):
	print '**********', msg
	df.cache()
	print df.rdd.getNumPartitions(), 'partitions'
	df.show(count, flag)
	
	if old != None:
		old.unpersist()




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
		#ngrams_thresh.unpersist()					

	#end training on different sample sizes		

	return models



def main(f \
		, outDir
		):

	
	#read file
	df = ReadFile(f, spark, nPartLog)

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

		outFile = outputFile(outDir, ctxSize)
		ngrams.write.csv(outFile)
		
		sqc.clearCache()
	#end train and test for each context size


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


catSizeLog = 10








main(f, outDir)

sys.exit()


