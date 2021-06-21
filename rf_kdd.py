from pyspark.sql.session import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import FloatType
from pyspark.sql.types import StringType
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import regexp_replace
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import datetime

#### spark session ####
spark = SparkSession.builder.master("local[*]").appName("RF_Calssification").getOrCreate()

#### build the schema to load dataframe ####
columnames = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
      "hot", "num_failed_logins", "logged_in", "num_compromised",
      "root_shell", "su_attempted", "num_root", "num_file_creations",
      "num_shells", "num_access_files", "num_outbound_cmds",
      "is_host_login", "is_guest_login", "count", "srv_count",
      "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
      "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
      "dst_host_count", "dst_host_srv_count",
      "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
      "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
      "dst_host_serror_rate", "dst_host_srv_serror_rate",
      "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "LABEL"]

schemadata = {}
schemadata[columnames[0]] = FloatType()
for i in range(1, 4):
    schemadata[columnames[i]] = StringType()
for i in range(4, 41):
    schemadata[columnames[i]] = FloatType()
schemadata[columnames[41]] = StringType()
schema = StructType([StructField(key, value, False) for key, value in schemadata.items()])

#### loading data ####
assembler = VectorAssembler(inputCols=columnames[:-1], outputCol="features")
training = spark.read.csv("C:/Users/asus/Desktop/AUT/BD/t2/data/aa.txt", schema = schema)
test=spark.read.csv("C:/Users/asus/Desktop/AUT/BD/t2/data/test.txt", schema = schema)

# Convert non numeric attributs to numeric for testing dataset
indexers = {}
for i in range(1, 4):
    name = columnames[i]
    indexers[name] = StringIndexer(inputCol=name, outputCol="_"+name, handleInvalid='keep')
    indexers[name] = indexers[name].fit(training)
    training = indexers[name].transform(training)
    training = training.drop(name)
    training = training.withColumnRenamed("_"+name, name)

#training.printSchema()
training = assembler.transform(training) 
def correct_labels(data):
    data = data.withColumn('LABEL', regexp_replace('LABEL', 'ftp_write.', 'R2L'))
    data = data.withColumn('LABEL', regexp_replace('LABEL', 'guess_passwd.', 'R2L'))
    data = data.withColumn('LABEL', regexp_replace('LABEL', 'imap.', 'R2L'))
    data = data.withColumn('LABEL', regexp_replace('LABEL', 'multihop.', 'R2L'))
    data = data.withColumn('LABEL', regexp_replace('LABEL', 'phf.', 'R2L'))
    data = data.withColumn('LABEL', regexp_replace('LABEL', 'spy.', 'R2L'))
    data = data.withColumn('LABEL', regexp_replace('LABEL', 'warezclient.', 'R2L'))
    data = data.withColumn('LABEL', regexp_replace('LABEL', 'warezmaster.', 'R2L'))
    data = data.withColumn('LABEL', regexp_replace('LABEL', 'buffer_overflow.', 'U2R'))
    data = data.withColumn('LABEL', regexp_replace('LABEL', 'loadmodule.', 'U2R'))
    data = data.withColumn('LABEL', regexp_replace('LABEL', 'perl.', 'U2R'))
    data = data.withColumn('LABEL', regexp_replace('LABEL', 'rootkit.', 'U2R'))
    data = data.withColumn('LABEL', regexp_replace('LABEL', 'back.', 'DOS'))
    data = data.withColumn('LABEL', regexp_replace('LABEL', 'land.', 'DOS'))
    data = data.withColumn('LABEL', regexp_replace('LABEL', 'neptune.', 'DOS'))
    data = data.withColumn('LABEL', regexp_replace('LABEL', 'pod.', 'DOS'))
    data = data.withColumn('LABEL', regexp_replace('LABEL', 'smurf.', 'DOS'))
    data = data.withColumn('LABEL', regexp_replace('LABEL', 'teardrop.', 'DOS'))
    data = data.withColumn('LABEL', regexp_replace('LABEL', 'ipsweep.', 'Probe'))
    data = data.withColumn('LABEL', regexp_replace('LABEL', 'nmap.', 'Probe'))
    data = data.withColumn('LABEL', regexp_replace('LABEL', 'portsweep.', 'Probe'))
    data = data.withColumn('LABEL', regexp_replace('LABEL', 'satan.', 'Probe'))
    data = data.withColumn('LABEL', regexp_replace('LABEL', 'normal.', 'Normal'))
    return data

training=correct_labels(training)
test=correct_labels(test)

#### seting a label for unkown labels of test data ####
from pyspark.sql.functions import when
label_list=['Normal','Probe','DOS','U2R','R2L']
test = test.withColumn("LABEL", when((test["LABEL"] != 'Normal')\
                                     & (test["LABEL"] !='Probe') & (test["LABEL"] !='DOS')\
                                     & (test["LABEL"] !='U2R') & (test["LABEL"] !='R2L') , 'Non').otherwise(test["LABEL"]))

#### preparing data ####
LabelIndexer = StringIndexer(inputCol='LABEL', outputCol="label" , handleInvalid='keep').fit(training)
training=LabelIndexer.transform(training)
training_data = training.select("features", "label")
training_data.show()

test_data=LabelIndexer.transform(test)
for i in range(1, 4):
    name = columnames[i]
    test_data=indexers[name].transform(test_data)
    test_data = test_data.drop(name)
    test_data = test_data.withColumnRenamed("_"+name, name) 
test_data = assembler.transform(test_data) 
test_data = test_data.select("features", "label")

#### random forest model ####

rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=40, maxBins=80 , maxDepth=20 , impurity='gini' )

t0 = datetime.datetime.now()   
model = rf.fit(training_data)
t= datetime.datetime.now()
running_time=int((t-t0).total_seconds())
print(running_time)
predictions = model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(accuracy)



predictionAndLabels=predictions.select('prediction','label')
predictionAndLabels=predictionAndLabels.rdd
from pyspark.mllib.evaluation import MulticlassMetrics
metrics = MulticlassMetrics(predictionAndLabels)
#### Overall statistics ####
TP=metrics.truePositiveRate(1.0)
FP=metrics.falsePositiveRate(1.0)
precision = metrics.precision(1.0)
recall = metrics.recall(1.0)
f1Score = metrics.fMeasure(1.0)






