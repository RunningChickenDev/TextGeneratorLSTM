import betternn as nn
import datetime
import time

# info it needs:
# - input file
# - input weights
# - output file

def write(infile, inweights, outfile, diversities=[0.5, 0.7, 1.0], infout=True):
	nn.Data.model = nn.load_model(nn.Data.vals, infile, inweights, model_type='double')

	cqw = 0
	curr_paragraph = ""
	paragraphs = []
	with open(infile, 'r') as f:
		for line in f:
			if line == '\n':
				# nn.w.msg("Another line found!", "Writer")
				cqw += 1
			else:
				# nn.w.msg_val("Analyzing line", line, "Writer")
				if cqw == 1:	# one enter!
					paragraphs += [str(curr_paragraph)]	# new object (?)
					curr_paragraph = ""
				cqw = 0
				curr_paragraph += line

	nn.w.msg_val("Paragraph count", len(paragraphs), "Writer")


	nn.w.msg("Writting file; this may take a while (go eat something) ...", "Writer")
	starttime = float(time.time())

	f = open(outfile, 'w+')
	f.write("========\n")
	f.write("Text written by 'writer.py' on {}\n".format(datetime.datetime.now()))
	f.write("--------\n")
	i = 0
	for p in paragraphs:
		nn.w.msg("Writing paragraph {} with length {} on {}".format(i, len(p), time.strftime('%X %x')), "Writer")
		if infout:
			f.write("Paragraph = {}\n".format(i))
		if infout:
			f.write("Paragraph Length = {}\n".format(len(p)))
		nn.generate(nn.Data.model, nn.Data.vals, f, diversities=[0.7], length=len(p), infout=infout)
		f.flush()
		i += 1

	endtime = float(time.time())

	nn.w.msg_val("Writing the text took (s)", endtime - starttime, "Writer")

	f.close()

if __name__ == '__main__':
	write('hmt/eses.txt', 'hmt/ww/weights-40-0.6914.hdf5', 'hmt/written1.min.txt', infout=False)
