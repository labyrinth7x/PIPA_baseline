shrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions

export JAVA_HOME=/usr/lib/jvm/java-1.7.0
export HADOOP_PREFIX=/usr/local/hadoop-2.7.2
export HADOOP_HOME=$HADOOP_PREFIX
export CLASSPATH=$HADOOP_PREFIX/lib/classpath_hdfs.jar
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HADOOP_PREFIX/lib/native:$JAVA_HOME/jre/lib/amd64/server
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64
export PATH=$PATH:/usr/local/cuda-8.0/bin
export PYTHONPATH=/mnt/data-7/data/qi01.zhang/incubator-mxnet/python


