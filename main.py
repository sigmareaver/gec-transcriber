import faulthandler
import os

import ctranslate2
import transformers
import argparse

from os import listdir
from os.path import isfile, isdir, join, split, splitext
# import pandas as pd
from bs4 import BeautifulSoup
import csv
from tqdm import tqdm
import nltk
import sys
import asyncio
from fastpunct import FastPunct
import glob
import shutil

ascii_map = [
    [0x00AB, '"'],
    [0x00AD, '-'],
    [0x00B4, '\''],
    [0x00BB, '"'],
    [0x00F7, '/'],
    [0x01C0, '|'],
    [0x01C3, '!'],
    [0x02B9, '\''],
    [0x02BA, '"'],
    [0x02BC, '\''],
    [0x02C4, '^'],
    [0x02C6, '^'],
    [0x02C8, '\''],
    [0x02CB, '`'],
    [0x02CD, '_'],
    [0x02DC, '~'],
    [0x0300, '`'],
    [0x0301, '\''],
    [0x0302, '^'],
    [0x0303, '~'],
    [0x030B, '"'],
    [0x030E, '"'],
    [0x0331, '_'],
    [0x0332, '_'],
    [0x0338, '/'],
    [0x0589, ':'],
    [0x05C0, '|'],
    [0x05C3, ':'],
    [0x066A, '%'],
    [0x066D, '*'],
    [0x200B, ' '],
    [0x2010, '-'],
    [0x2011, '-'],
    [0x2012, '-'],
    [0x2013, '-'],
    [0x2014, '-'],
    [0x2015, '-'],
    [0x2016, '|'],
    [0x2017, '_'],
    [0x2018, '\''],
    [0x2019, '\''],
    [0x201A, ','],
    [0x201B, '\''],
    [0x201C, '"'],
    [0x201D, '"'],
    [0x201E, '"'],
    [0x201F, '"'],
    [0x2032, '\''],
    [0x2033, '"'],
    [0x2034, '\''],
    [0x2035, '`'],
    [0x2036, '"'],
    [0x2037, '\''],
    [0x2038, '^'],
    [0x2039, '<'],
    [0x203A, '>'],
    [0x203D, '?'],
    [0x2044, '/'],
    [0x204E, '*'],
    [0x2052, '%'],
    [0x2053, '~'],
    [0x2060, ' '],
    [0x20E5, '\\'],
    [0x2212, '-'],
    [0x2215, '/'],
    [0x2216, '\\'],
    [0x2217, '*'],
    [0x2223, '|'],
    [0x2236, ':'],
    [0x223C, '~'],
    [0x2264, '<'],
    [0x2265, '>'],
    [0x2266, '<'],
    [0x2267, '>'],
    [0x2303, '^'],
    [0x2329, '<'],
    [0x232A, '>'],
    [0x266F, '#'],
    [0x2731, '*'],
    [0x00AB, '"'],
    [0x00AD, '-'],
    [0x00B4, '\''],
    [0x00BB, '"'],
    [0x00F7, '/'],
    [0x01C0, '|'],
    [0x01C3, '!'],
    [0x02B9, '\''],
    [0x02BA, '"'],
    [0x02BC, '\''],
    [0x02C4, '^'],
    [0x02C6, '^'],
    [0x02C8, '\''],
    [0x02CB, '`'],
    [0x02CD, '_'],
    [0x02DC, '~'],
    [0x0300, '`'],
    [0x0301, '\''],
    [0x0302, '^'],
    [0x0303, '~'],
    [0x030B, '"'],
    [0x030E, '"'],
    [0x0331, '_'],
    [0x0332, '_'],
    [0x0338, '/'],
    [0x0589, ':'],
    [0x05C0, '|'],
    [0x05C3, ':'],
    [0x066A, '%'],
    [0x066D, '*'],
    [0x200B, ' '],
    [0x2010, '-'],
    [0x2011, '-'],
    [0x2012, '-'],
    [0x2013, '-'],
    [0x2014, '-'],
    [0x2015, '-'],
    [0x2016, '|'],
    [0x2017, '_'],
    [0x2018, '\''],
    [0x2019, '\''],
    [0x201A, ','],
    [0x201B, '\''],
    [0x201C, '"'],
    [0x201D, '"'],
    [0x201E, '"'],
    [0x201F, '"'],
    [0x2032, '\''],
    [0x2033, '"'],
    [0x2034, '\''],
    [0x2035, '`'],
    [0x2036, '"'],
    [0x2037, '\''],
    [0x2038, '^'],
    [0x2039, '<'],
    [0x203A, '>'],
    [0x203D, '?'],
    [0x2044, '/'],
    [0x204E, '*'],
    [0x2052, '%'],
    [0x2053, '~'],
    [0x2060, ' '],
    [0x20E5, '\\'],
    [0x2212, '-'],
    [0x2215, '/'],
    [0x2216, '\\'],
    [0x2217, '*'],
    [0x2223, '|'],
    [0x2236, ':'],
    [0x223C, '~'],
    [0x2264, '<'],
    [0x2265, '>'],
    [0x2266, '<'],
    [0x2267, '>'],
    [0x2303, '^'],
    [0x2329, '<'],
    [0x2758, '|'],
    [0x2762, '!'],
    [0x27E6, '['],
    [0x27E8, '<'],
    [0x27E9, '>'],
    [0x2983, '{'],
    [0x2984, '}'],
    [0x3003, '"'],
    [0x3008, '<'],
    [0x3009, '>'],
    [0x301B, ']'],
    [0x301C, '~'],
    [0x301D, '"'],
    [0x301E, '"'],
    [0xFEFF, ' ']
]


def convert_unicode(text: str):
    tmp = ''
    for char in text:
        for i in range(len(ascii_map)):
            if char == ascii_map[i][0]:
                char = ascii_map[i][1]
                break
        if ord(char) < 128:
            tmp = tmp+''.join(char)
    return tmp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='GEC Transcriber',
        description='Transcribe datasets through a GEC model.',
        epilog=''
    )

    parser.add_argument('model_path', type=str,
                        help='model to load')

    subparsers = parser.add_subparsers(dest='mode')
    convert_p = subparsers.add_parser('convert', help='convert model')
    convert_p.add_argument('output', type=str, help='output model filename')
    convert_p.add_argument('-q', choices=['auto', 'int8', 'int8_float16', 'int16', 'float16'])

    transcribe_p = subparsers.add_parser('transcribe', help='process dataset')
    transcribe_p.add_argument('dataset', type=str, help='dataset directory or filename')
    transcribe_p.add_argument('-b', '--batch-size', type=int)
    transcribe_p.add_argument('-s', '--strip-html', action='store_true')
    transcribe_p.add_argument('-u', '--convert-unicode', action='store_true')
    transcribe_p.add_argument('-f', '--fastpunct', action='store_true')
    transcribe_p.add_argument('-d', '--device', choices=['cpu', 'cuda'])
    transcribe_p.add_argument('-ct', '--cpu-threads', type=int, help='# cpu threads')
    transcribe_p.add_argument('-gt', '--gpu-threads', type=int, help='# gpu threads')

    args = parser.parse_args()

    if args.mode == 'transcribe':
        if not hasattr(args, 'batch_size'):
            args.batch_size = 1
        if isdir(args.dataset):
            inPath = args.dataset
            outPath = join(args.transcribe, 'output/')
            datasets = [[], []]
            datasets[0] = [f for f in listdir(inPath) if isfile(join(inPath, f))]
            datasets[1] = [splitext(f)[0]+'.csv' for f in datasets[0]]
        elif isfile(args.dataset):
            inPath, filename = split(args.dataset)
            outPath = join(inPath, 'output/')
            datasets = [[], []]
            datasets[0] = [filename]
            datasets[1] = [splitext(filename)[0]+'.csv']
        else:
            raise "Dataset not found!"

        if not os.path.exists(outPath):
            os.makedirs(outPath)

        nltk.download('punkt')

        if hasattr(args, 'device'):
            device = args.device
        else:
            device = 'auto'

        translator = ctranslate2.Translator('models/'+args.model_path, device=device or 'auto',
                                            inter_threads=args.cpu_threads or 16, intra_threads=args.gpu_threads or 16, max_queued_batches=4)
        tokenizer = transformers.AutoTokenizer.from_pretrained('models/'+args.model_path)

        prompt = "Rewrite with proper spelling, grammar, and punctuation: "

        csv.field_size_limit(sys.maxsize)

        if args.fastpunct:
            fastpunct = FastPunct()

        with open(join(outPath, 'anomalies.txt'), 'w', newline='') as anomalies:
            empty = 0
            long = 0
            nonascii = 0
            bad = 0
            ok = 0
            poor = 0

            for i in range(len(datasets)):
                with open(join(inPath, datasets[0][i]), 'r', newline='') as infile, \
                          open(join(outPath, datasets[1][i]), 'w', newline='') as outfile:
                    ext = splitext(datasets[0][i])[1]
                    if ext == '.txt':
                        reader = csv.reader(infile, delimiter=':', skipinitialspace=True, doublequote=True,
                                            escapechar='\\', quotechar='\'', quoting=csv.QUOTE_ALL)
                    elif ext == '.csv':
                        reader = csv.reader(infile, delimiter=',', skipinitialspace=True, doublequote=True,
                                            escapechar='\\', quotechar='\"', quoting=csv.QUOTE_ALL)
                    else:
                        continue
                    print("Processing: " + datasets[0][i] + '\n')
                    writer = csv.writer(outfile, delimiter=',', doublequote=False,
                                        escapechar='\\', quotechar='\'', quoting=csv.QUOTE_ALL)

                    data = list(reader)
                    num_cols = len(data[0])

                    for j in tqdm(range(0, len(data), args.batch_size)):
                        if ext == '.txt':
                            for k in range(args.batch_size):
                                row = data[j + k]
                                if len(row) > num_cols:
                                    for x in range(len(row) - 1, 1, -1):
                                        if x == 1:
                                            break
                                        row[x - 1] = row[x - 1] + ": " + row[x]
                                        row.pop(x)
                                if len(row) == 0:
                                    anomalies.write("EMPTY ROW: " + datasets[0][i] + ": " + str(reader.line_num) + '\n')
                                    empty += 1
                                    continue
                                if len(row) == 1:
                                    if len(row[0]) > 0:
                                        if len(row[0]) > 2048:
                                            anomalies.write("TOO LONG: " + datasets[1][i] + ": " +
                                                            str(reader.line_num) + ": " + row[0] + '\n')
                                            long += 1
                                            continue
                                        elif not row[0].isascii() or (len(row) > 1 and not row[1].isascii()):
                                            anomalies.write("NON-ASCII: " + datasets[1][i] + ": " +
                                                            str(reader.line_num) + ": " + row[0] + '\n')
                                            nonascii += 1
                                            continue
                                        elif not row[0][0].isalnum() and (row[0][0] != '*' and row[0][0] != '>'):
                                            anomalies.write("Probably BAD: " + datasets[1][i] + ": " +
                                                            str(reader.line_num) + ": " + row[0] + '\n')
                                            bad += 1
                                            continue
                                        elif len(row[0]) > 4 and row[0].startswith('*...') and row[0][4].isalnum():
                                            ok += 1
                                        elif (len(row[0]) > 1 and row[0][0] == '*' and (
                                                not row[0][1].isalnum() and not row[0][1] == '(' and
                                                not row[0][1] == '"' and not row[0][1] == '\'' and
                                                not row[0][1] == '*')):
                                            anomalies.write("May contain poor quality data: " + datasets[1][i] + ": " +
                                                            str(reader.line_num) + ": " + row[0] + '\n')
                                            poor += 1
                                            continue

                        if args.strip_html:
                            y = 5
                            for x in range(args.batch_size):
                                # for y in range(len(data[j + x])):
                                soup = BeautifulSoup(data[j + x][y], 'html.parser')
                                for br in soup.find_all('br'):
                                    br.replace_with(' ')
                                plain_text = soup.get_text()
                                data[j + x][y] = plain_text

                        if args.convert_unicode:
                            for x in range(args.batch_size):
                                for y in range(len(data[j + x])):
                                    data[j + x][y] = convert_unicode(data[j + x][y])

                        # for i in range(args.batch_size):
                        sentences = []
                        max_len = []

                        for x in range(args.batch_size):
                            sentences.append([])
                            for y in range(len(data[j + x])):
                                sentences[x].append([])
                                sentences[x][y] = nltk.tokenize.sent_tokenize(str(data[j + x][y]))
                                # print(f"{x} {y}")
                                if args.fastpunct and y == 5:
                                    sentences[x][y] = fastpunct.punct(sentences[x][y], correct=True)

                        for y in range(len(data[j])):
                            max_len.append(0)
                            max_len[y] = max([len(sentences[x][y]) for x in range(args.batch_size)])
                            # if len(sentences[x][y]) > max_len[y]:
                            #     max_len[y] = len(sentences[x][y])

                        for y in range(len(data[j])):
                            if y == 5:
                                joined_results = []
                                for z in range(max_len[y]):
                                    ids = []
                                    input_tokens = []
                                    for x in range(args.batch_size):
                                        # print(f"{x} {y} {z}")
                                        joined_results.append("")
                                        try:
                                            if len(sentences[x][y]) > z:
                                                ids.append(x)
                                                # input_tokens.append([])
                                                # s = sentences[x][y][z]
                                                tmp = tokenizer.encode(prompt + sentences[x][y][z])
                                                tmp2 = tokenizer.convert_ids_to_tokens(tmp)
                                                input_tokens.append(tmp2)
                                        except:
                                            print(sentences)

                                    async def translate():
                                        async_results = []
                                        # joined_results = []
                                        async_results.extend(translator.translate_batch(input_tokens,
                                                             max_batch_size=args.batch_size,
                                                             batch_type='examples', asynchronous=True,
                                                             max_input_length=512))
                                        for a in range(len(async_results)):
                                            id = ids[a]
                                            # joined_results.append("")
                                            r = async_results[a].result()
                                            output = ""
                                            for h in r.hypotheses:
                                                output_tokens = h  # results[0].hypotheses[0]
                                                output = str(tokenizer.decode(tokenizer.convert_tokens_to_ids(output_tokens)))
                                            try:
                                                if not joined_results[id]:
                                                    joined_results[id] = output
                                                else:
                                                    joined_results[id] = joined_results[id] + ' ' + output
                                            except:
                                                print('caught exception')

                                        # print(joined_results[0])
                                        # writer.writerow(data[j + x])
                                        # return joined_results

                                    asyncio.run(translate())
                                    del input_tokens
                                    del ids

                                for x in range(len(joined_results)):
                                    if len(joined_results[x]) == 0:
                                        continue
                                    data[j + x][y] = joined_results[x]
                                del joined_results

                        for x in range(args.batch_size):
                            writer.writerow(data[j + x])

                        # end of for row in reader

                    infile.close()
                    outfile.close()
                    # end of with open()
                # end of for datasets loop
            anomalies.write("Empty Rows: " + str(empty) + "   Too Long: " + str(long) + "   Non-ASCII: " +
                            str(nonascii) + "   BAD: " + str(bad) + "   Poor Quality: " + str(poor) + '\n')
            anomalies.close()
            print("Done!")
        # if hasattr(args, 'device'):
        #     device = args.device
        # else:
        #     device = 'auto'
        # translator = ctranslate2.Translator('models/'+args.model_path, device=device or 'auto')
        # tokenizer = transformers.AutoTokenizer.from_pretrained('../full-models/'+args.model_path)
        #
        # prompt = "Rewrite this food sentence to sound proper: "
        # input_text = "I got a grate idea for a new desert."
        # print('Input: ' + prompt + input_text + '\n')
        #
        # input_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt + input_text))
        # results = translator.translate_batch([input_tokens])
        #
        # for r in results:
        #     for h in r.hypotheses:
        #         output_tokens = h #results[0].hypotheses[0]
        #         output = tokenizer.decode(tokenizer.convert_tokens_to_ids(output_tokens))
        #         print('Output: ' + output + '\n')
    elif args.mode == 'convert':
        converter = ctranslate2.converters.TransformersConverter(args.model_path)
        converter.convert(args.output, quantization=args.q or 'auto', force=True)
        for file in glob.glob(args.model_path + "*.json"):
            shutil.copy(file, args.output)
        for file in glob.glob(args.model_path + "*.model"):
            shutil.copy(file, args.output)
    else:
        parser.error('specify -c or -t')
        parser.print_help()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
