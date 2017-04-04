import json
import sys

import requests
import time
from mrjob.job import MRJob
from mrjob.protocol import RawValueProtocol

sys.path.append('../')


class IndexNSFW(MRJob):
    INPUT_PROTOCOL = RawValueProtocol
    OUTPUT_PROTOCOL = RawValueProtocol

    HADOOP_INPUT_FORMAT = 'org.apache.hadoop.mapred.lib.NLineInputFormat'

    JOBCONF = {'mapreduce.task.timeout': '9600000',
               'mapreduce.input.fileinputformat.split.maxsize': '50000000',
               'mapreduce.map.speculative': 'false',
               'mapreduce.reduce.speculative': 'false',
               'mapreduce.job.jvm.numtasks': '-1',
               'mapreduce.input.lineinputformat.linespermap': 100,
               'mapred.job.priority': 'VERY_HIGH'
               }


    def mapper(self, _, line):
        #line = line.split('\t')[1]
        json_doc = json.loads(line)
        json_data = {"image": json_doc['srcBase64']}

        retries_count = 0
        success = False
        while not success and retries_count < 5:
            try:
                response = requests.post('http://p29.arquivo.pt:9080/safeimage', json=json_data)
                json_doc['NSFW'] = json.loads(response.content)['NSFW']
                json_str = json.dumps(json_doc)
                self.stdout.write(json_str + '\n')
                success = True
            except requests.ConnectionError:
                retries_count += 1
                time.sleep(10)
        if not success:
            self.stderr.write('Unable to process the line: {}'.format(line))

if __name__ == "__main__":
    IndexNSFW.run()
