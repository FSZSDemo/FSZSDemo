# coding=utf-8
import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataFile',
                        type=str,
                        help='path to dataset')
    parser.add_argument('--generated_dataFile',
                        type=str,
                        help='path to generated dataset')
    parser.add_argument('--label_dataFile',
                        type=str,
                        help='path to labels')
    parser.add_argument('--fileVocab',
                        type=str,
                        help='path to pretrained model vocab')
    parser.add_argument('--fileModelConfig',
                        type=str,
                        help='path to pretrained model config')

    parser.add_argument('--fileModel',
                        type=str,
                        help='path to pretrained model')

    parser.add_argument('--fileModelSave',
                        type=str,
                        help='path to save model')
    

    parser.add_argument('--epochs',
                        type=int,
                        help='number of epochs to train',
                        default=100)
    parser.add_argument('--numNWay',
                        type=int,
                        help='number of classes per episode',
                        default=5)
    parser.add_argument('--numKShot',
                        type=int,
                        help='number of instances per class',
                        default=5)

    parser.add_argument('--numQShot',
                        type=int,
                        help='number of querys per class',
                        default=25)
    
    parser.add_argument('--episodeTrain',
                        type=int,
                        help='number of tasks per epoch in training process',
                        default=100)

    parser.add_argument('--episodeTest',
                        type=int,
                        help='number of tasks per epoch in testing process',
                        default=1000)

    parser.add_argument('--learning_rate',
                        type=float,
                        help='learning rate for the model, default=0.00001',
                        default=0.00001)



    parser.add_argument('--numDevice',
                        type=int,
                        help='id of gpu ',
                        default=0)

    # parser.add_argument('--warmup_steps',
    #                     type=int,
    #                     help='num of warmup_steps',
    #                     default=100)
    #
    # parser.add_argument('--weight_decay',
    #                     type=float,
    #                     help='ratio of decay',
    #                     default=0.2)
    #
    # parser.add_argument('--dropout_rate',
    #                     type=float,
    #                     help='ratio of dropout',
    #                     default=0.1)

    parser.add_argument('--hold_num',
                        type=int,
                        help='num of anchors per class ',
                        default=30)
    parser.add_argument('--label_num',
                        type=int,
                        help='num of labels of all unseen data ',
                        default=9)
    # parser.add_argument('--sample',
    #                     type=int,
    #                     help='num of generated samples per shot',
    #                     default=20)


    #-------
    parser.add_argument("--seed",
                        default=12,
                        type=int,
                        help="Random seed")
    parser.add_argument("--bert_model",
                        default='bert-base-uncased', #'roberta-base',
                        type=str,
                        help="BERT model")
    parser.add_argument("--train_batch_size",
                        default=370,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    # parser.add_argument("--learning_rate",
    #                     default=1e-5,
    #                     type=float,
    #                     help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=30,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--max_grad_norm', help='gradient clipping for Max gradient norm.', required=False, default=1.0,
                        type=float)
    parser.add_argument('--label_smoothing',
                        type=float,
                        default=0.1,
                        help='Coefficient for label smoothing (default: 0.1, if 0.0, no label smoothing)')
    parser.add_argument('--max_seq_length',
                        type=int,
                        default=128,
                        help='Maximum number of paraphrases for each sentence')
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lowercase input string")

    # Special params
    parser.add_argument('--train_file_path',
                        type=str,
                        default=None,
                        help='Training data path')
    parser.add_argument('--dev_file_path',
                        type=str,
                        default=None,
                        help='Validation data path')
    parser.add_argument('--oos_dev_file_path',
                        type=str,
                        default=None,
                        help='Out-of-Scope validation data path')

    parser.add_argument('--output_dir',
                        type=str,
                        default=None,
                        help='Output file path')
    parser.add_argument('--save_model_path',
                        type=str,
                        default='',
                        help='path to save the model checkpoints')

    parser.add_argument('--bert_nli_path',
                        type=str,
                        default='',
                        help='The bert checkpoints which are fine-tuned with NLI datasets')

    parser.add_argument("--scratch",
                        action='store_true',
                        help="Whether to start from the original BERT")

    parser.add_argument('--over_sampling',
                        type=int,
                        default=0,
                        help='Over-sampling positive examples as there are more negative examples')

    parser.add_argument('--few_shot_num',
                        type=int,
                        default=5,
                        help='Number of training examples for each class')
    # parser.add_argument('--num_trials',
    #                     type=int,
    #                     default=10,
    #                     help='Number of trials to see robustness')

    parser.add_argument("--do_predict",
                        action='store_true',
                        default=False,
                        help="do_predict the model")
    parser.add_argument("--do_final_test",
                        action='store_true',
                        help="do_predict the model")

    return parser
