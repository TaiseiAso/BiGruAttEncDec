# Attention Encoder-Decoder Dialog Model with GRU

<img width="500" alt="Model Architecture Image" src="https://user-images.githubusercontent.com/38200445/92203968-5e818e80-eebd-11ea-968c-e214eea47b5f.jpg">

## Goal
- generic and easy method using knowledge graph
- improve information, diversity, appropriateness
- reform decoding method and objective function

## Example
|Type|Sentence|
|:---:|:---|
|Post|so instead of **adding** a **simple** **key** combination, they make you use their **backwards** **mouse** .|
|Reference|no there is a dedicated button for it on the **keyboard** , or you can just keep your hand on your **input** **device** .|
|Beam Search|you 're right . i 'm not a smart guy , but i do n't think it 's necessary to be a **mouse**|
|+ Proposed|i do n't know if you can use a **mouse** **pad** , but it 's not **easy** to control the **mouse** .|

|Method|BLEU-1|BLEU-2|DIST-1|DIST-2|
|:---:|:---:|:---:|:---:|:---:|
|Beam Search|**14.545**|**2.827**|3.328|12.872|
|+ Proposed|14.285|2.771|**3.934**|**17.619**|

## Dataset
- use dataset for CCM
(Commonsense Knowledge Aware Conversation Generation with Graph Attention)  
- Commonsense Conversation Dataset (English)
- embedding vector is GloVe
- dialog data from Reddit (not Twitter)
- commonsense is ConceptNet

## Environment
- Windows10 (coding and run), Ubuntu18.04 (run)
- python 3.6.8
- pytorch 1.1.0
- CUDA 10.1

## Run
1. download CCM dataset
2. create folders (./data and ./log and ./model)
3. dataset into ./data folder
4. edit param.py
5. $ python train.py
6. (stop learning with 'Ctrl+C')
7. $ python test.py
8. $ pip install rouge (for ROUGE eval)
9. $ nltk.download('wordnet') (for METEOR eval)
10. $ python eval.py

## Implement
### model
- [x] prepare data
- [x] create dictionary
- [x] GRU encoder
- [x] GRU decoder
- [x] attention

### decoding method
- [x] greedy search
- [x] sampling
- [x] top-k sampling
- [x] top-p sampling
- [x] beam search
- [x] diverse beam search
- [x] maximum mutual information
- [x] \+ with knowledge graph

### objective function
- [x] softmax cross-entropy loss
- [x] inverse token frequency loss
- [x] inverse N-gram frequency loss
- [x] \+ with knowledge graph

### reranking method
- [x] beam score
- [ ] entity score
- [ ] breakdown possibility
- [ ] auto evaluation

### automatic evaluation
- [x] Length
- [x] BLEU
- [x] NIST
- [x] ROUGE
- [x] DIST
- [x] Repeat
- [x] METEOR
- [x] Entity score

### others
- [x] length normalization
- [x] repetitive suppression
- [x] IDF

## Reference (random order)
### Site
- http://coai.cs.tsinghua.edu.cn/hml/dataset/#commonsense
- https://conceptnet.io/
- https://huggingface.co/blog/how-to-generate
- https://miyanetdev.com/archives/1308
- https://qiita.com/m__k/items/646044788c5f94eadc8d
- https://takoroy-ai.hatenadiary.jp/entry/2018/07/02/224216
- https://github.com/jojonki/arXivNotes/issues/159
- https://stanford.edu/~shervine/l/ja/teaching/cs-230/cheatsheet-recurrent-neural-networks
- https://ksksksks2.hatenadiary.jp/entry/20191202/1575212640
- http://unicorn.ike.tottori-u.ac.jp/2010/s072046/paper/graduation-thesis/node32.html
- https://www.nltk.org/api/nltk.translate.html
- http://unicorn.ike.tottori-u.ac.jp/2012/s092013/paper/graduation-thesis/node31.html
- http://yagami12.hatenablog.com/entry/2017/12/30/175113
- https://medium.com/@aiii/pytorch-e64c248ab428
- https://qdata.github.io/deep2Read//talks2019/Extra19s/TkachStochasticBeamSearch.pdf

### Paper
- http://coai.cs.tsinghua.edu.cn/hml/media/files/2018_commonsense_ZhouHao_3_TYVQ7Iq.pdf
- https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/P6-12.pdf
- https://arxiv.org/pdf/1911.02707.pdf
- https://www.anlp.jp/proceedings/annual_meeting/2019/pdf_dir/F5-2.pdf
- https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/P5-11.pdf
- https://www.anlp.jp/proceedings/annual_meeting/2019/pdf_dir/P3-34.pdf
- https://arxiv.org/pdf/1904.09751.pdf
- https://ahcweb01.naist.jp/papers/conference/2019/201906_SIGNL_sara-as/201906_SIGNL_sara-as.paper.pdf
- https://www.jstage.jst.go.jp/article/jnlp/21/3/21_421/_pdf
- https://arxiv.org/pdf/1911.03587.pdf
- https://arxiv.org/pdf/1611.08562.pdf
- https://www.anlp.jp/proceedings/annual_meeting/2006/pdf_dir/B4-4.pdf
- https://anlp.jp/proceedings/annual_meeting/2018/pdf_dir/A4-2.pdf
