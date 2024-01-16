start_i=$1
end_i=$2
enable_kill=1
# -lt is less than operator
echo seed start $start_i, end $end_i
#Iterate the loop until a less than or equal 10
while [ $start_i -le $end_i ]
do
echo seed$start_i
if [ $enable_kill -eq 0 ]
then
    ps -ef |grep -E "python.*seed "$start_i" " | grep -v grep
else
    ps -ef |grep -E "python.*seed "$start_i" " | grep -v grep|awk '{print $2}'|xargs kill
fi
start_i=`expr $start_i + 1`
done