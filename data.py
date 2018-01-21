import json

class MetaData:
	def __init__(self):
		self.name = ""
		self.source = ""
		self.authors = []

def prepareData(inputtext, outputdata, outputmetadata):
	extractData(inputtext, outputdata)
	extractMetaData(inputtext, outputdata, outputmetadata, cast=cast)

def extractData(inputtext, outputdata):
	raw_text = ""
	f_in = open(inputtext, "r")
	f_out = open(outputdata, "w")
	for line in f_in.readlines():
		for c in line:
			if ord(c) < 128:
				f_out.write(c.lower())
	f_in.close()
	f_out.close()

def extractMetaData(inputtext, outputdata, outputmetadata, name="", authors=[]):
	obj = {}
	obj['name'] = name
	obj['source'] = inputtext
	obj['data'] = outputdata
	obj['authors'] = authors

	with open(outputmetadata, 'w') as f:
		json.dump(obj, f, indent=4)

def generateGenericMetaData(outputmetadata):
	obj = {}
	obj['name'] = "DataSet"
	obj['authors'] = ["Generic User"]
	obj['source'] = "dataset.txt"
	obj['data'] = "dataset.data.txt"

	obj['char_to_int'] = {}
	obj['int_to_char'] = {}
	for i in range(0, 26):
		obj['char_to_int'][chr(97+i)] = i
		obj['int_to_char'][i] = chr(97+i)

	with open(outputmetadata, 'w') as f:
		json.dump(obj, f, sort_keys=True, indent=4)


if __name__ == '__main__':
	generateGenericMetaData("generic.data.json")
	# prepareData("dataset_bnw_en.data.txt", "bnw/bnw.data.txt", "bnw/bnw.data.json")
	# extractMetaData("BNW chp. 1-6", "bnw/bnw.data.json", "../dataset_bnw_en.data.txt")
