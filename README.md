# Attention Encoder-Decoder Dialog Model with GRU

## Goal
- generic and easy method using knowledge graph
- improve information, diversity, appropriateness
- reform decoding method (and objective function)

## Dataset
- use dataset for CCM
(Commonsense Knowledge Aware Conversation Generation with Graph Attention)  
- Commonsense Conversation Dataset (English)
- embedding vector is GloVe
- dialog data from Reddit (not Twitter)
- commonsense is ConceptNet

## Environment
- python 3.6.8
- pytorch 1.1.0
- CUDA 10.1

## Implement
### model
- [x] prepare data
- [x] create dictionary
- [x] GRU encoder
- [x] GRU decoder
- [x] attention

### decoding method
- [x] greedy search
- [ ] sampling
- [ ] top-k sampling
- [ ] top-p sampling
- [ ] beam search
- [ ] diverse beam search
- [ ] maximum mutual information
- [ ] \+ with knowledge graph

### objective function
- [x] softmax cross-entropy loss
- [ ] \+ with knowledge graph

### reranking method
- [ ] beam score
- [ ] entity score

### automatic evaluation
- [ ] length
- [ ] BLEU
- [ ] NIST
- [ ] ROUGE
- [ ] dist
- [ ] METEOR
- [ ] entity score

### advance
- [ ] repeat suppressor
- [ ] inverse token frequency loss
- [ ] inverse N-gram frequency loss
- [ ] reranking by breakdown possibility

## Site
- http://coai.cs.tsinghua.edu.cn/hml/dataset/#commonsense
- https://conceptnet.io/
- https://huggingface.co/blog/how-to-generate
- https://miyanetdev.com/archives/1308
- https://qiita.com/m__k/items/646044788c5f94eadc8d
- https://takoroy-ai.hatenadiary.jp/entry/2018/07/02/224216

## Paper
- http://coai.cs.tsinghua.edu.cn/hml/media/files/2018_commonsense_ZhouHao_3_TYVQ7Iq.pdf
- https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/P6-12.pdf
- https://arxiv.org/pdf/1911.02707.pdf
- https://www.anlp.jp/proceedings/annual_meeting/2019/pdf_dir/F5-2.pdf
- https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/P5-11.pdf
- https://www.anlp.jp/proceedings/annual_meeting/2019/pdf_dir/P3-34.pdf
- https://arxiv.org/pdf/1904.09751.pdf
