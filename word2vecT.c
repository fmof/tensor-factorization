// TODO: add total word count to vocabulary, instead of "train_words"
//
// Modifed by Frank Ferraro, March-April 2016
// Added:
//   - support for dynamic tensor dimensions
//
/////////////////////////////////////////////////////////////////
//
//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "vocab.h"
#include "io.h"

extern int IO_PRINT;
extern int EOS_INDEX;

#define MAX_STRING 1000
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

typedef float real;                    // Precision of float numbers

char train_file[MAX_STRING], output_file[MAX_STRING];
char dumpvec_file[MAX_STRING];
int binary = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 1, min_reduce = 1, use_position = 0;
long long layer1_size = 100;
long long train_words = 0, word_count_actual = 0, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 0;
real *expTable;
real **syn;
clock_t start;
int numiters = 1;
int tensor = 1;

int tensor_dim = 3;

struct vocabulary** vocabs;

int negative = 15;
const int table_size = 1e8;
int *unitable;

long long GetFileSize(char *fname) {
  long long fsize;
  FILE *fin = fopen(fname, "rb");
  if (fin == NULL) {
    printf("ERROR: file not found! %s\n", fname);
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  fsize = ftell(fin);
  fclose(fin);
  return fsize;
}


// Used for sampling of negative examples.
// wc[i] == the count of context number i
// wclen is the number of entries in wc (context vocab size)
void InitUnigramTable(struct vocabulary *v) {
  int a, i;
  long long normalizer = 0;
  real d1, power = 0.75;
  unitable = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < v->vocab_size; a++) normalizer += pow(v->vocab[a].cn, power);
  i = 0;
  d1 = pow(v->vocab[i].cn, power) / (real)normalizer;
  for (a = 0; a < table_size; a++) {
    unitable[a] = i;
    if (a / (real)table_size > d1) {
      i++;
      d1 += pow(v->vocab[i].cn, power) / (real)normalizer;
    }
    if (i >= v->vocab_size) i = v->vocab_size - 1;
  }
}

void InitNet() {
   long long a, b;
   // initialize word vectors
   size_t i = 0;
   syn = (real**)malloc(tensor_dim * sizeof(real*));
   //a = posix_memalign((void **)&, 128, (long long)vocabs[i]->vocab_size * layer1_size * sizeof(real));
   for(; i < 2 && i < tensor_dim; ++i) {
     syn[i] = (real*)malloc(sizeof(real) * (long long)vocabs[i]->vocab_size * layer1_size );
     if (syn[i] == NULL) {printf("Memory allocation failed\n"); exit(1);}
     for (b = 0; b < layer1_size; b++) {
       for (a = 0; a < vocabs[i]->vocab_size; a++) {
	 syn[i][a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
       }
     }
   }
   // initialize relation vectors
   //a = posix_memalign((void **)&syn2, 128, (long long)vocabs[i]->vocab_size * layer1_size * sizeof(real));
   for(; i < tensor_dim; ++i) {
     syn[i] = (real*)malloc(sizeof(real) * (long long)vocabs[i]->vocab_size * layer1_size );
     if (syn[i] == NULL) {printf("Memory allocation failed\n"); exit(1);}
     for (b = 0; b < layer1_size; b++) {
       for (a = 0; a < vocabs[i]->vocab_size; a++) {
	 syn[i][a * layer1_size + b] = 1.0 ; //(rand() / (real)RAND_MAX - 0.5) / layer1_size;
       }
     }
   }
}

// Read word,context pairs from training file, where both word and context are integers.
// We are learning to predict context based on word.
//
// Word and context come from different vocabularies, but we do not really care about that
// at this point.
void *TrainModelThread(void *id) {
  int* word_indices = (int*)malloc(tensor_dim * sizeof(int));
  long long* offsets = (long long*)malloc(tensor_dim * sizeof(long long));
  int ii = 0;
  for(ii = 0; ii < tensor_dim; ++ii) {
    word_indices[ii] = -1;
    offsets[ii] = 0;
  }
  long long d;
  long long word_count = 0, last_word_count = 0;
  long long c, target, label;
  unsigned long long next_random = (unsigned long long)id;
  real f, g;
  clock_t now;
  real *neu = (real*)malloc(layer1_size * tensor_dim * sizeof(real));
  FILE *fi = fopen(train_file, "rb");
  long long start_offset = file_size / (long long)num_threads * (long long)id;
  long long end_offset = file_size / (long long)num_threads * (long long)(id+1);
  int iter;
  //printf("thread %d %lld %lld \n",id, start_offset, end_offset);
  for (iter=0; iter < numiters; ++iter) {
     fseek(fi, start_offset, SEEK_SET);
     // if not binary:
     while (fgetc(fi) != '\n') { }; // TODO make sure its ok
     printf("thread %d %ld\n", id, ftell(fi));
     
     long long train_words = vocabs[0]->word_count;
     while (1) { //HERE @@@
        // TODO set alpha scheduling based on number of examples read.
        // The conceptual change is the move from word_count to pair_count
        if (word_count - last_word_count > 10000) {
           word_count_actual += word_count - last_word_count;
           last_word_count = word_count;
           if ((debug_mode > 1)) {
              now=clock();
              printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
                    word_count_actual / (real)(numiters*train_words + 1) * 100,
                    word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
              fflush(stdout);
           }
           alpha = starting_alpha * (1 - word_count_actual / (real)(numiters*train_words + 1));
           if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
        }
	
        if (feof(fi) || ftell(fi) > end_offset) {
	  break;
	}
	
        for (c = 0; c < tensor_dim * layer1_size; c++) {
	  neu[c] = 0.0;
	}

	// get word indices
	int any_neg = 0;
	for(ii = 0; ii < tensor_dim; ++ii) {
	  word_indices[ii] = ReadWordIndex(vocabs[ii], fi);
	  offsets[ii] = word_indices[ii] * layer1_size;
	  if(word_indices[ii] < 0) any_neg = 1;
	  if(IO_PRINT) printf("%d\n", word_indices[ii]);
	}

        word_count++; //TODO ?
        if (any_neg) {
	  continue;
	}

        // NEGATIVE SAMPLING
        for (d = 0; d < negative + 1; d++) {
           if (d == 0) {
              target = word_indices[1];
              label = 1;
           } else {
              next_random = next_random * (unsigned long long)25214903917 + 11;
              target = unitable[(next_random >> 16) % table_size];
              if (target == 0) {
		target = next_random % (vocabs[1]->vocab_size - 1) + 1;
	      }
              if (target == word_indices[1]) {
		continue;
	      }
              label = 0;
           }
	   
           offsets[1] = target * layer1_size;

	   // gets the function value
           f = 0;
	   for (c = 0; c < layer1_size; c++) {
	     real finner = 1.0;
	     for(ii = 0; ii < tensor_dim; ++ii) {
	       finner *= syn[ii][c + offsets[ii]];
	     }
	     f += finner;
	   }

	   if (f > MAX_EXP) {
	     g = (label - 1) * alpha;
	   } else if (f < -MAX_EXP) {
	     g = (label - 0) * alpha;
	   } else {
	     g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
	   }

	   for(ii = 0; ii < tensor_dim; ++ii) {
	     if(1 == ii) continue;
	     const size_t ii_off = ii * layer1_size;	     
	     for (c = 0; c < layer1_size; c++) {
	       int jj;
	       real f_ = 1.0;
	       for(jj = 0; jj < tensor_dim; ++jj) {
		 if(ii == jj) continue;
		 f_ *= syn[jj][offsets[jj] + c];
	       }
	       neu[ii_off + c] += g * f_;
	     }
	   }
	   for (c = 0; c < layer1_size; c++) {
	     int jj;
	     real f_ = 1.0;
	     for(jj = 0; jj < tensor_dim; ++jj) {
	       if(1 == jj) continue;
	       f_ *= syn[jj][offsets[jj] + c];
	     }
	     syn[1][c + offsets[1]] += g * f_;
	   }
        } // end negative sampling
        // Learn weights input -> hidden
	for(ii = 0; ii < tensor_dim; ++ii) {
	  if(1 == ii) continue;
	  real* const syn_ = syn[ii];
	  const long long offset = offsets[ii];
	  const size_t noff = ii * layer1_size;
	  for (c = 0; c < layer1_size; c++) {
	    syn_[c + offset] += neu[noff + c];
	  }
	}
     }
  }
  fclose(fi);
  free(neu);
  free(word_indices);
  free(offsets);
  pthread_exit(NULL);
}

void read_vocabs() {
  FILE *fi = fopen(train_file, "rb");
  printf("Accumulating vocabulary...");
  fflush(stdout);
  while (fgetc(fi) != '\n') { }; // TODO make sure its ok

  unsigned int ii;
  vocabs = (struct vocabulary**)malloc(tensor_dim * sizeof(struct vocabulary*));
  for(ii = 0; ii < tensor_dim; ++ii) {
    vocabs[ii] = CreateVocabulary();
  }

  char word[1024];
  long long word_idx;
  int* highest_seen_words = (int*)malloc(tensor_dim * sizeof(int));
  for(ii = 0; ii < tensor_dim; ++ii) {
    highest_seen_words[ii] = -1;
  }
  unsigned long long line_num = 1;
  while (1) {
    if (feof(fi)) {
      break;
    }
    IO_PRINT=0;
    for(ii = 0; ii < tensor_dim; ++ii) {
      ReadWord(word, fi, 1024);
      if (feof(fi)) break;
      word_idx = ProvisionalAddWordToVocab(vocabs[ii], word);
      if(word_idx > highest_seen_words[ii]) {
	highest_seen_words[ii] = word_idx;
	vocabs[ii]->vocab[word_idx].cn = 0;
      }
      vocabs[ii]->vocab[word_idx].cn += 1;
    }
    ++line_num;
  }
  //IO_PRINT=1;
  printf(" reducing vocab...");
  fflush(stdout);
  for(ii = 0; ii < tensor_dim; ++ii) {
    SortAndReduceVocab(vocabs[ii], 0);
  }
  fclose(fi);
  free(highest_seen_words);
  printf(" done\n");
  for(ii = 0; ii < tensor_dim; ++ii) {
    printf("Vocab %d size: %ld\n", ii, vocabs[ii]->vocab_size);
    printf("Word %d count: %lld\n", ii, vocabs[ii]->word_count);
  }
  fflush(stdout);
}

void TrainModel() {
  long a, b;
  FILE *fo, *fo2;
  file_size = GetFileSize(train_file);
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;

  fo = fopen(train_file, "rb");
  tensor_dim = -1;
  while(1) {
    if(feof(fo)) break;
    int ch = 0, nw = 0, insp = 0;
    do {
      ch = fgetc(fo);
      if((ch == ' ') | (ch == '\t')) {
	if(insp == 0) {
	  ++nw;
	}
	insp = 1;
      } else if(ch == '\n') {
	++nw;
	break;
      } else {
	insp = 0;
      }
    } while(1);
    if(nw > 0) tensor_dim = nw;
    break;
  }
  fclose(fo);
  if(tensor_dim < 0) {
    printf("ERROR reading in columns\n");
    exit(1);
  }
  printf("Expecting %d columns\n", tensor_dim);

  // we need to set EOS_INDEX to -1 in order to properly sort the entire vocabulary
  EOS_INDEX = -1;
  read_vocabs();
  
  InitNet();
  struct vocabulary* wv = vocabs[0];
  InitUnigramTable(vocabs[1]);

  start = clock();
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  printf("\n");
  fo = fopen(output_file, "wb");
  if (classes == 0) {
    // Save the context word vectors
    if(dumpvec_file[0] != 0) {
      int ii;
      for(ii = 1; ii < tensor_dim; ++ii) {
	// "col_XXX" ==> 7 + 1 null
	char d_name[MAX_STRING + 8];
	size_t ci = 0;
	while(ci < MAX_STRING + 8) d_name[ci++] = 0;
	ci = 0;
	while(ci < MAX_STRING && dumpvec_file[ci] != 0) {
	  d_name[ci] = dumpvec_file[ci];
	  ++ci;
	}
	const char* colstr = "col_";
	strncpy(d_name + ci, colstr, 4);
	ci += 4;
	//itoa(ii, d_name + ci, 10);
	sprintf(d_name + ci, "%d", ii);
	printf("writing vectors for column-%d to %s\n", ii, d_name);
	fo2 = fopen(d_name, "wb");
        fprintf(fo2, "%d %d\n", vocabs[ii]->vocab_size, layer1_size);
        for (a = 0; a < vocabs[ii]->vocab_size; a++) {
	  fprintf(fo2, "%s ", vocabs[ii]->vocab[a].word);
	  if (binary) {
	    for (b = 0; b < layer1_size; b++) {
	      fwrite(&syn[ii][a * layer1_size + b], sizeof(real), 1, fo2);
	    }
	  } else {
	    for (b = 0; b < layer1_size; b++)
	      fprintf(fo2, "%lf ", syn[ii][a * layer1_size + b]);
	  }
	  fprintf(fo2, "\n");
	}
	fclose(fo2);
      }
    }
    /* if (dumpcv_file[0] != 0) { */

    /* } */
    /* // Save relation vectors */
    /* if (dumprv_file[0] != 0) { */
    /*     fo2 = fopen(dumprv_file, "wb"); */
    /*     fprintf(fo2, "%d %d\n", rv->vocab_size, layer1_size); */
    /*     for (a = 0; a < rv->vocab_size; a++) { */
    /*        fprintf(fo2, "%s ", rv->vocab[a].word); //TODO */
    /*        if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn2[a * layer1_size + b], sizeof(real), 1, fo2); */
    /*        else for (b = 0; b < layer1_size; b++) fprintf(fo2, "%lf ", syn2[a * layer1_size + b]); */
    /*        fprintf(fo2, "\n"); */
    /*    } */
    /* } */
    
    fprintf(fo, "%ld %lld\n", wv->vocab_size, layer1_size);
    for (a = 0; a < wv->vocab_size; a++) {
      fprintf(fo, "%s ", wv->vocab[a].word); //TODO
      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn[0][a * layer1_size + b], sizeof(real), 1, fo);
      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn[0][a * layer1_size + b]);
      fprintf(fo, "\n");
    }
  } else {
    // Run K-means on the word vectors
    /* int clcn = classes, iter = 10, closeid; */
    /* int *centcn = (int *)malloc(classes * sizeof(int)); */
    /* int *cl = (int *)calloc(wv->vocab_size, sizeof(int)); */
    /* real closev, x; */
    /* real *cent = (real *)calloc(classes * layer1_size, sizeof(real)); */
    /* for (a = 0; a < wv->vocab_size; a++) cl[a] = a % clcn; */
    /* for (a = 0; a < iter; a++) { */
    /*    printf("kmeans iter %d\n", a); */
    /*   for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0; */
    /*   for (b = 0; b < clcn; b++) centcn[b] = 1; */
    /*   for (c = 0; c < wv->vocab_size; c++) { */
    /*     for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d]; */
    /*     centcn[cl[c]]++; */
    /*   } */
    /*   for (b = 0; b < clcn; b++) { */
    /*     closev = 0; */
    /*     for (c = 0; c < layer1_size; c++) { */
    /*       cent[layer1_size * b + c] /= centcn[b]; */
    /*       closev += cent[layer1_size * b + c] * cent[layer1_size * b + c]; */
    /*     } */
    /*     closev = sqrt(closev); */
    /*     for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev; */
    /*   } */
    /*   for (c = 0; c < wv->vocab_size; c++) { */
    /*     closev = -10; */
    /*     closeid = 0; */
    /*     for (d = 0; d < clcn; d++) { */
    /*       x = 0; */
    /*       for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b]; */
    /*       if (x > closev) { */
    /*         closev = x; */
    /*         closeid = d; */
    /*       } */
    /*     } */
    /*     cl[c] = closeid; */
    /*   } */
    /* } */
    /* // Save the K-means classes */
    /* for (a = 0; a < wv->vocab_size; a++) fprintf(fo, "%s %d\n", wv->vocab[a].word, cl[a]); */
    /* free(centcn); */
    /* free(cent); */
    /* free(cl); */
  }
  fclose(fo);
  int ii;
  for(ii = 0; ii < tensor_dim; ++ii) {
    free(vocabs[ii]);
  }
  free(vocabs);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1b\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 15, common values are 5 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 1)\n");
    //printf("\t-min-count <int>\n");
    //printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words and contexts. Those that appear with higher frequency");
    printf(" in the training data will be randomly down-sampled; default is 0 (off), useful value in the original word2vec was 1e-5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025\n");
    printf("\t-iters <int>\n");
    printf("\t\tPerform i iterations over the data; default is 1\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-dumpvec filename\n");
    printf("\t\tDump all context vectors in their own file, with prefix <filename>\n");
    printf("\nExamples:\n");
    printf("./word2vecf -train data.txt -output vec.txt -size 200 -negative 5 -threads 10 \n\n");
    return 0;
  }
  output_file[0] = 0;
  dumpvec_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-dumpv", argc, argv)) > 0) strcpy(dumpvec_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-iters", argc, argv)) > 0) numiters = atoi(argv[i+1]);
  if ((i = ArgPos((char *)"-tensor", argc, argv)) > 0) tensor = atoi(argv[i+1]);
  printf("tensor: %d\n", tensor);
  
  if (output_file[0] == 0) { printf("must supply -output.\n\n"); return 0; }
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  free(expTable);
  return 0;
}
