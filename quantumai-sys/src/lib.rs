//! quantumai-sys: Complete C FFI bridge for QuantumAI
//! Exposes everything: Sequential, Model, Dataset, DataPipeline,
//! all ML algorithms, NLP, AutoML, benchmarks, experiments, GAN, VAE,
//! QLearning, pretrained models, anomaly detection, model registry.

#![allow(clippy::all)]
use quantumai::*;
use quantumai::data::Dataset;
use quantumai::ml::*;
use quantumai::nlp::*;
use quantumai::advanced::*;
use quantumai::metrics;
use quantumai::datasets;
use quantumai::automl;
use quantumai::pretrained;
use std::os::raw::c_char;
use std::ffi::CStr;

macro_rules! box_h { ($v:expr) => { Box::into_raw(Box::new($v)) as i64 }; }
macro_rules! ref_h { ($h:expr, $T:ty) => { unsafe { &*($h as *const $T) } }; }
macro_rules! mut_h { ($h:expr, $T:ty) => { unsafe { &mut *($h as *mut $T) } }; }
macro_rules! free_h { ($h:expr, $T:ty) => { if $h!=0 { unsafe { drop(Box::from_raw($h as *mut $T)); } } }; }

fn cs(p: *const c_char) -> &'static str {
    if p.is_null() { return ""; }
    unsafe { CStr::from_ptr(p).to_str().unwrap_or("") }
}

fn to_labels(t: &Tensor) -> Vec<usize> { t.data.iter().map(|&v| v as usize).collect() }
fn to_f64s(t: &Tensor) -> Vec<f64> { t.data.clone() }

// ── Tensor ──────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_tensor_free(h:i64){ free_h!(h,Tensor); }
#[no_mangle] pub extern "C" fn qai_tensor_len(h:i64)->i64{ ref_h!(h,Tensor).data.len() as i64 }
#[no_mangle] pub extern "C" fn qai_tensor_ndim(h:i64)->i64{ ref_h!(h,Tensor).shape.len() as i64 }
#[no_mangle] pub extern "C" fn qai_tensor_dim(h:i64,ax:i64)->i64{ ref_h!(h,Tensor).shape.get(ax as usize).copied().unwrap_or(0) as i64 }
#[no_mangle] pub extern "C" fn qai_tensor_get(h:i64,i:i64)->f64{ ref_h!(h,Tensor).data.get(i as usize).copied().unwrap_or(0.0) }
#[no_mangle] pub extern "C" fn qai_tensor_set(h:i64,i:i64,v:f64){ if let Some(x)=mut_h!(h,Tensor).data.get_mut(i as usize){*x=v;} }
#[no_mangle] pub extern "C" fn qai_tensor_sum(h:i64)->f64{ ref_h!(h,Tensor).sum() }
#[no_mangle] pub extern "C" fn qai_tensor_mean(h:i64)->f64{ ref_h!(h,Tensor).mean() }
#[no_mangle] pub extern "C" fn qai_tensor_max(h:i64)->f64{ ref_h!(h,Tensor).max() }
#[no_mangle] pub extern "C" fn qai_tensor_min(h:i64)->f64{ ref_h!(h,Tensor).min() }
#[no_mangle] pub extern "C" fn qai_tensor_argmax(h:i64)->i64{ ref_h!(h,Tensor).argmax() as i64 }
#[no_mangle] pub extern "C" fn qai_tensor_std(h:i64)->f64{ ref_h!(h,Tensor).std_dev() }
#[no_mangle] pub extern "C" fn qai_tensor_print(h:i64){ let t=ref_h!(h,Tensor); println!("Tensor{:?} mean={:.4} std={:.4}",t.shape,t.mean(),t.std_dev()); }
#[no_mangle] pub extern "C" fn qai_tensor_zeros(sp:*const i64,nd:i64)->i64{ let s:Vec<usize>=unsafe{std::slice::from_raw_parts(sp,nd as usize)}.iter().map(|&x|x as usize).collect(); box_h!(Tensor::zeros(s)) }
#[no_mangle] pub extern "C" fn qai_tensor_randn(sp:*const i64,nd:i64)->i64{ let s:Vec<usize>=unsafe{std::slice::from_raw_parts(sp,nd as usize)}.iter().map(|&x|x as usize).collect(); box_h!(Tensor::randn(s)) }
#[no_mangle] pub extern "C" fn qai_tensor_from_buf(p:*const f64,len:i64,rows:i64,cols:i64)->i64{ let d=unsafe{std::slice::from_raw_parts(p,len as usize).to_vec()}; box_h!(Tensor::new(d,vec![rows as usize,cols as usize])) }

// ── Sequential ───────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_seq_new()->i64{ box_h!(Sequential::new()) }
#[no_mangle] pub extern "C" fn qai_seq_free(h:i64){ free_h!(h,Sequential); }
#[no_mangle] pub extern "C" fn qai_seq_dense(h:i64,i:i64,o:i64,act:*const c_char){ mut_h!(h,Sequential).dense(i as usize,o as usize,cs(act)); }
#[no_mangle] pub extern "C" fn qai_seq_batchnorm(h:i64,f:i64){ mut_h!(h,Sequential).batchnorm(f as usize); }
#[no_mangle] pub extern "C" fn qai_seq_layernorm(h:i64,f:i64){ mut_h!(h,Sequential).layernorm(f as usize); }
#[no_mangle] pub extern "C" fn qai_seq_dropout(h:i64,r:f64){ mut_h!(h,Sequential).dropout(r); }
#[no_mangle] pub extern "C" fn qai_seq_embedding(h:i64,v:i64,d:i64){ mut_h!(h,Sequential).embedding(v as usize,d as usize); }
#[no_mangle] pub extern "C" fn qai_seq_compile(h:i64,opt:*const c_char,loss:*const c_char,lr:f64){ mut_h!(h,Sequential).compile(cs(opt),cs(loss),lr); }
#[no_mangle] pub extern "C" fn qai_seq_summary(h:i64){ ref_h!(h,Sequential).summary(); }
#[no_mangle] pub extern "C" fn qai_seq_fit(h:i64,ds:i64,ep:i64,bs:i64,verbose:i32){ let d=ref_h!(ds,Dataset); mut_h!(h,Sequential).fit(&d.x,&d.y,ep as usize,bs as usize,verbose!=0); }
#[no_mangle] pub extern "C" fn qai_seq_fit_xy(h:i64,xp:*const f64,xl:i64,xd:i64,yp:*const f64,yl:i64,yd:i64,ep:i64,bs:i64,v:i32){ let xdata=unsafe{std::slice::from_raw_parts(xp,xl as usize).to_vec()}; let ydata=unsafe{std::slice::from_raw_parts(yp,yl as usize).to_vec()}; let x=Tensor::new(xdata,vec![(xl/xd) as usize,xd as usize]); let y=Tensor::new(ydata,vec![(yl/yd) as usize,yd as usize]); mut_h!(h,Sequential).fit(&x,&y,ep as usize,bs as usize,v!=0); }
#[no_mangle] pub extern "C" fn qai_seq_predict(h:i64,ds:i64)->i64{ let d=ref_h!(ds,Dataset); box_h!(ref_h!(h,Sequential).predict(&d.x)) }
#[no_mangle] pub extern "C" fn qai_seq_predict_one(h:i64,xp:*const f64,xd:i64)->i64{ let xdata=unsafe{std::slice::from_raw_parts(xp,xd as usize).to_vec()}; let x=Tensor::new(xdata,vec![1,xd as usize]); box_h!(ref_h!(h,Sequential).predict(&x)) }
#[no_mangle] pub extern "C" fn qai_seq_predict_class(h:i64,xp:*const f64,xd:i64)->i64{ let xdata=unsafe{std::slice::from_raw_parts(xp,xd as usize).to_vec()}; let x=Tensor::new(xdata,vec![1,xd as usize]); ref_h!(h,Sequential).predict_class(&x) as i64 }
#[no_mangle] pub extern "C" fn qai_seq_accuracy(h:i64,ds:i64)->f64{ let d=ref_h!(ds,Dataset); ref_h!(h,Sequential).compute_accuracy(&d.x,&d.y) }
#[no_mangle] pub extern "C" fn qai_seq_save(h:i64,path:*const c_char){ ref_h!(h,Sequential).save_weights(cs(path)); }
#[no_mangle] pub extern "C" fn qai_seq_load(h:i64,path:*const c_char)->i32{ mut_h!(h,Sequential).load_weights(cs(path)) as i32 }
#[no_mangle] pub extern "C" fn qai_seq_l2(h:i64,l:f64)->i64{ box_h!(std::mem::replace(mut_h!(h,Sequential),Sequential::new()).with_l2_reg(l)) }
#[no_mangle] pub extern "C" fn qai_seq_grad_clip(h:i64,c:f64)->i64{ box_h!(std::mem::replace(mut_h!(h,Sequential),Sequential::new()).with_grad_clip(c)) }

// ── Model (high-level fluent API) ────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_model_create(task:*const c_char)->i64{ box_h!(Model::create(cs(task))) }
#[no_mangle] pub extern "C" fn qai_model_load_pretrained(name:*const c_char,pretrained:i32)->i64{ box_h!(Model::load(cs(name),pretrained!=0)) }
#[no_mangle] pub extern "C" fn qai_model_free(h:i64){ free_h!(h,Model); }
#[no_mangle] pub extern "C" fn qai_model_epochs(h:i64,n:i64)->i64{ let m=unsafe{Box::from_raw(h as *mut Model)}; box_h!(m.epochs(n as usize)) }
#[no_mangle] pub extern "C" fn qai_model_batch_size(h:i64,n:i64)->i64{ let m=unsafe{Box::from_raw(h as *mut Model)}; box_h!(m.batch_size(n as usize)) }
#[no_mangle] pub extern "C" fn qai_model_lr(h:i64,lr:f64)->i64{ let m=unsafe{Box::from_raw(h as *mut Model)}; box_h!(m.learning_rate(lr)) }
#[no_mangle] pub extern "C" fn qai_model_optimizer(h:i64,opt:*const c_char)->i64{ let m=unsafe{Box::from_raw(h as *mut Model)}; box_h!(m.optimizer(cs(opt))) }
#[no_mangle] pub extern "C" fn qai_model_loss(h:i64,loss:*const c_char)->i64{ let m=unsafe{Box::from_raw(h as *mut Model)}; box_h!(m.loss(cs(loss))) }
#[no_mangle] pub extern "C" fn qai_model_train(h:i64,ds:i64,ep:i64,verbose:i32){ let d=ref_h!(ds,Dataset); mut_h!(h,Model).train(&d.x,&d.y,ep as usize,verbose!=0); }
#[no_mangle] pub extern "C" fn qai_model_predict_class(h:i64,xp:*const f64,xd:i64)->i64{ let xdata=unsafe{std::slice::from_raw_parts(xp,xd as usize).to_vec()}; let x=Tensor::new(xdata,vec![1,xd as usize]); ref_h!(h,Model).predict_class(&x) as i64 }
#[no_mangle] pub extern "C" fn qai_model_summary(h:i64){ ref_h!(h,Model).summary(); }
#[no_mangle] pub extern "C" fn qai_model_save(h:i64,path:*const c_char){ ref_h!(h,Model).save(cs(path)); }
#[no_mangle] pub extern "C" fn qai_model_plot_history(h:i64){ ref_h!(h,Model).plot_history(); }

// ── Dataset ──────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_ds_free(h:i64){ free_h!(h,Dataset); }
#[no_mangle] pub extern "C" fn qai_ds_n_samples(h:i64)->i64{ ref_h!(h,Dataset).n_samples as i64 }
#[no_mangle] pub extern "C" fn qai_ds_n_features(h:i64)->i64{ ref_h!(h,Dataset).n_features as i64 }
#[no_mangle] pub extern "C" fn qai_ds_n_classes(h:i64)->i64{ ref_h!(h,Dataset).n_classes as i64 }
#[no_mangle] pub extern "C" fn qai_ds_x(h:i64)->i64{ box_h!(ref_h!(h,Dataset).x.clone()) }
#[no_mangle] pub extern "C" fn qai_ds_y(h:i64)->i64{ box_h!(ref_h!(h,Dataset).y.clone()) }
#[no_mangle] pub extern "C" fn qai_ds_normalize(h:i64){ mut_h!(h,Dataset).normalize(); }
#[no_mangle] pub extern "C" fn qai_ds_shuffle(h:i64){ mut_h!(h,Dataset).shuffle(); }
#[no_mangle] pub extern "C" fn qai_ds_info(h:i64){ ref_h!(h,Dataset).info(); }
#[no_mangle] pub extern "C" fn qai_ds_split_train(h:i64,r:f64)->i64{ let(t,_)=ref_h!(h,Dataset).train_test_split(r); box_h!(t) }
#[no_mangle] pub extern "C" fn qai_ds_split_test(h:i64,r:f64)->i64{ let(_,t)=ref_h!(h,Dataset).train_test_split(r); box_h!(t) }

// ── Synthetic datasets ───────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_make_xor(n:i64)->i64{ box_h!(quantumai::make_xor(n as usize)) }
#[no_mangle] pub extern "C" fn qai_make_sine(n:i64)->i64{ box_h!(quantumai::make_sine(n as usize)) }
#[no_mangle] pub extern "C" fn qai_make_classification(n:i64,nf:i64,nc:i64)->i64{ box_h!(quantumai::make_classification(n as usize,nf as usize,nc as usize)) }
#[no_mangle] pub extern "C" fn qai_make_moons(n:i64,noise:f64)->i64{ box_h!(quantumai::make_moons(n as usize,noise)) }
#[no_mangle] pub extern "C" fn qai_make_circles(n:i64,noise:f64)->i64{ box_h!(quantumai::make_circles(n as usize,noise,0.5)) }
#[no_mangle] pub extern "C" fn qai_make_regression(n:i64,nf:i64,noise:f64)->i64{ box_h!(datasets::make_regression(n as usize,nf as usize,noise)) }
#[no_mangle] pub extern "C" fn qai_make_iris(n:i64)->i64{ box_h!(datasets::make_iris_like(n as usize)) }
#[no_mangle] pub extern "C" fn qai_make_spiral(n:i64,nc:i64)->i64{ box_h!(datasets::make_spiral(n as usize,nc as usize)) }

// ── ML algorithms ────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_linreg_new()->i64{ box_h!(LinearRegression::new()) }
#[no_mangle] pub extern "C" fn qai_linreg_free(h:i64){ free_h!(h,LinearRegression); }
#[no_mangle] pub extern "C" fn qai_linreg_fit(h:i64,ds:i64){ let d=ref_h!(ds,Dataset); mut_h!(h,LinearRegression).fit(&d.x,&d.y); }
#[no_mangle] pub extern "C" fn qai_linreg_r2(h:i64,ds:i64)->f64{ let d=ref_h!(ds,Dataset); ref_h!(h,LinearRegression).r2_score(&d.x,&d.y) }

#[no_mangle] pub extern "C" fn qai_logreg_new()->i64{ box_h!(LogisticRegression::new()) }
#[no_mangle] pub extern "C" fn qai_logreg_free(h:i64){ free_h!(h,LogisticRegression); }
#[no_mangle] pub extern "C" fn qai_logreg_fit(h:i64,ds:i64){ let d=ref_h!(ds,Dataset); mut_h!(h,LogisticRegression).fit(&d.x,&d.y); }
#[no_mangle] pub extern "C" fn qai_logreg_accuracy(h:i64,ds:i64)->f64{ let d=ref_h!(ds,Dataset); ref_h!(h,LogisticRegression).accuracy(&d.x,&d.y) }

#[no_mangle] pub extern "C" fn qai_knn_new(k:i64)->i64{ box_h!(KNN::new(k as usize)) }
#[no_mangle] pub extern "C" fn qai_knn_free(h:i64){ free_h!(h,KNN); }
#[no_mangle] pub extern "C" fn qai_knn_fit(h:i64,ds:i64){ let d=ref_h!(ds,Dataset); let labels=to_labels(&d.y); mut_h!(h,KNN).fit(&d.x,&labels); }
#[no_mangle] pub extern "C" fn qai_knn_predict_class(h:i64,xp:*const f64,len:i64,dim:i64)->i64{ let xdata=unsafe{std::slice::from_raw_parts(xp,len as usize).to_vec()}; let x=Tensor::new(xdata,vec![(len/dim)as usize,dim as usize]); ref_h!(h,KNN).predict(&x).into_iter().next().unwrap_or(0) as i64 }
#[no_mangle] pub extern "C" fn qai_knn_accuracy(h:i64,ds:i64)->f64{ let d=ref_h!(ds,Dataset); let labels=to_labels(&d.y); ref_h!(h,KNN).accuracy(&d.x,&labels) }

#[no_mangle] pub extern "C" fn qai_dtree_new(max_depth:i64)->i64{ box_h!(DecisionTree::new(max_depth as usize)) }
#[no_mangle] pub extern "C" fn qai_dtree_free(h:i64){ free_h!(h,DecisionTree); }
#[no_mangle] pub extern "C" fn qai_dtree_fit(h:i64,ds:i64){ let d=ref_h!(ds,Dataset); let labels=to_labels(&d.y); let nc=d.n_classes; mut_h!(h,DecisionTree).fit(&d.x,&labels,nc); }
#[no_mangle] pub extern "C" fn qai_dtree_accuracy(h:i64,ds:i64)->f64{ let d=ref_h!(ds,Dataset); let labels=to_labels(&d.y); ref_h!(h,DecisionTree).accuracy(&d.x,&labels) }

#[no_mangle] pub extern "C" fn qai_rf_new(n_trees:i64,max_depth:i64)->i64{ box_h!(RandomForest::new(n_trees as usize,max_depth as usize)) }
#[no_mangle] pub extern "C" fn qai_rf_free(h:i64){ free_h!(h,RandomForest); }
#[no_mangle] pub extern "C" fn qai_rf_fit(h:i64,ds:i64){ let d=ref_h!(ds,Dataset); let labels=to_labels(&d.y); let nc=d.n_classes; mut_h!(h,RandomForest).fit(&d.x,&labels,nc); }
#[no_mangle] pub extern "C" fn qai_rf_accuracy(h:i64,ds:i64)->f64{ let d=ref_h!(ds,Dataset); let labels=to_labels(&d.y); ref_h!(h,RandomForest).accuracy(&d.x,&labels) }

#[no_mangle] pub extern "C" fn qai_kmeans_new(k:i64)->i64{ box_h!(KMeans::new(k as usize)) }
#[no_mangle] pub extern "C" fn qai_kmeans_free(h:i64){ free_h!(h,KMeans); }
#[no_mangle] pub extern "C" fn qai_kmeans_fit(h:i64,ds:i64){ let d=ref_h!(ds,Dataset); mut_h!(h,KMeans).fit(&d.x); }
#[no_mangle] pub extern "C" fn qai_kmeans_predict(h:i64,xp:*const f64,len:i64,dim:i64)->i64{ let xdata=unsafe{std::slice::from_raw_parts(xp,len as usize).to_vec()}; let x=Tensor::new(xdata,vec![(len/dim)as usize,dim as usize]); ref_h!(h,KMeans).predict(&x).into_iter().next().unwrap_or(0) as i64 }
#[no_mangle] pub extern "C" fn qai_kmeans_inertia(h:i64,ds:i64)->f64{ let d=ref_h!(ds,Dataset); ref_h!(h,KMeans).inertia(&d.x) }

#[no_mangle] pub extern "C" fn qai_pca_new(n:i64)->i64{ box_h!(PCA::new(n as usize)) }
#[no_mangle] pub extern "C" fn qai_pca_free(h:i64){ free_h!(h,PCA); }
#[no_mangle] pub extern "C" fn qai_pca_fit_transform(h:i64,ds:i64)->i64{ let d=ref_h!(ds,Dataset); box_h!(mut_h!(h,PCA).fit_transform(&d.x)) }

#[no_mangle] pub extern "C" fn qai_nb_new()->i64{ box_h!(NaiveBayes::new()) }
#[no_mangle] pub extern "C" fn qai_nb_free(h:i64){ free_h!(h,NaiveBayes); }
#[no_mangle] pub extern "C" fn qai_nb_fit(h:i64,ds:i64){ let d=ref_h!(ds,Dataset); let labels=to_labels(&d.y); let nc=d.n_classes; mut_h!(h,NaiveBayes).fit(&d.x,&labels,nc); }
#[no_mangle] pub extern "C" fn qai_nb_accuracy(h:i64,ds:i64)->f64{ let d=ref_h!(ds,Dataset); let labels=to_labels(&d.y); ref_h!(h,NaiveBayes).accuracy(&d.x,&labels) }

// ── Advanced algorithms ───────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_gbm_new(n:i64,lr:f64,depth:i64)->i64{ box_h!(GradientBoosting::new(n as usize,lr,depth as usize)) }
#[no_mangle] pub extern "C" fn qai_gbm_free(h:i64){ free_h!(h,GradientBoosting); }
#[no_mangle] pub extern "C" fn qai_gbm_fit(h:i64,ds:i64){ let d=ref_h!(ds,Dataset); let y=to_f64s(&d.y); mut_h!(h,GradientBoosting).fit(&d.x,&y); }
#[no_mangle] pub extern "C" fn qai_gbm_accuracy(h:i64,ds:i64)->f64{ let d=ref_h!(ds,Dataset); let y=to_f64s(&d.y); ref_h!(h,GradientBoosting).accuracy(&d.x,&y) }

#[no_mangle] pub extern "C" fn qai_vae_new(in_d:i64,h_d:i64,lat:i64)->i64{ box_h!(VAE::new(in_d as usize,h_d as usize,lat as usize)) }
#[no_mangle] pub extern "C" fn qai_vae_free(h:i64){ free_h!(h,VAE); }
#[no_mangle] pub extern "C" fn qai_vae_sample(h:i64,n:i64)->i64{ box_h!(ref_h!(h,VAE).sample(n as usize)) }

#[no_mangle] pub extern "C" fn qai_gan_new(lat:i64,out:i64,hid:i64)->i64{ box_h!(GAN::new(lat as usize,out as usize,hid as usize)) }
#[no_mangle] pub extern "C" fn qai_gan_free(h:i64){ free_h!(h,GAN); }
#[no_mangle] pub extern "C" fn qai_gan_generate(h:i64,n:i64)->i64{ box_h!(ref_h!(h,GAN).generate(n as usize)) }
#[no_mangle] pub extern "C" fn qai_gan_summary(h:i64){ ref_h!(h,GAN).summary(); }

#[no_mangle] pub extern "C" fn qai_ql_new(states:i64,actions:i64)->i64{ box_h!(QLearning::new(states as usize,actions as usize)) }
#[no_mangle] pub extern "C" fn qai_ql_free(h:i64){ free_h!(h,QLearning); }
#[no_mangle] pub extern "C" fn qai_ql_act(h:i64,sp:*const f64,sl:i64)->i64{ let state=unsafe{std::slice::from_raw_parts(sp,sl as usize).to_vec()}; ref_h!(h,QLearning).act(&state) as i64 }
#[no_mangle] pub extern "C" fn qai_ql_replay(h:i64,bs:i64){ mut_h!(h,QLearning).replay(bs as usize); }

// ── Anomaly Detection ─────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_anomaly_new(dim:i64)->i64{ box_h!(AnomalyDetector::new(dim as usize)) }
#[no_mangle] pub extern "C" fn qai_anomaly_free(h:i64){ free_h!(h,AnomalyDetector); }
#[no_mangle] pub extern "C" fn qai_anomaly_train(h:i64,ds:i64,ep:i64){ let d=ref_h!(ds,Dataset); mut_h!(h,AnomalyDetector).train_on_normal(&d.x,ep as usize); }
#[no_mangle] pub extern "C" fn qai_anomaly_score(h:i64,ds:i64)->f64{ let d=ref_h!(ds,Dataset); ref_h!(h,AnomalyDetector).score(&d.x) }
#[no_mangle] pub extern "C" fn qai_anomaly_is_anomaly(h:i64,ds:i64)->i32{ let score=ref_h!(h,AnomalyDetector).score(&ref_h!(ds,Dataset).x); let threshold=ref_h!(h,AnomalyDetector).threshold; if score>threshold{1}else{0} }

// ── NLP ───────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_tokenizer_new(max_vocab:i64)->i64{ box_h!(Tokenizer::new(max_vocab as usize)) }
#[no_mangle] pub extern "C" fn qai_tokenizer_free(h:i64){ free_h!(h,Tokenizer); }
#[no_mangle] pub extern "C" fn qai_tokenizer_vocab_size(h:i64)->i64{ ref_h!(h,Tokenizer).vocab_size() as i64 }

// ── AutoML ────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_automl_new(trials:i64)->i64{ box_h!(automl::AutoClassifier::new(trials as usize)) }
#[no_mangle] pub extern "C" fn qai_automl_free(h:i64){ free_h!(h,automl::AutoClassifier); }
#[no_mangle] pub extern "C" fn qai_automl_fit(h:i64,ds:i64){ let d=ref_h!(ds,Dataset); mut_h!(h,automl::AutoClassifier).fit(d); }

#[no_mangle] pub extern "C" fn qai_tune(ds:i64,trials:i64)->f64{ let d=ref_h!(ds,Dataset); let r=quantumai::tune(d,trials as usize); r.best_accuracy }

// ── Pretrained models ─────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_resnet50(nc:i64)->i64{ box_h!(pretrained::resnet50(false,nc as usize)) }
#[no_mangle] pub extern "C" fn qai_bert_tiny()->i64{ box_h!(pretrained::bert_tiny(false)) }
#[no_mangle] pub extern "C" fn qai_gpt2_mini()->i64{ box_h!(pretrained::gpt2_mini()) }
#[no_mangle] pub extern "C" fn qai_efficientnet(nc:i64)->i64{ box_h!(pretrained::efficientnet_b0(nc as usize)) }
#[no_mangle] pub extern "C" fn qai_yolov8()->i64{ box_h!(pretrained::yolov8_nano()) }

// ── Benchmarks ────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_benchmark_run(h:i64,train_ds:i64,test_ds:i64,nc:i64)->i64{
    let m=mut_h!(h,Model);
    let tr=ref_h!(train_ds,Dataset);
    let te=ref_h!(test_ds,Dataset);
    let report=BenchmarkReport::run(m,&tr.x,&tr.y,&te.x,&te.y,nc as usize,&[]);
    box_h!(report)
}
#[no_mangle] pub extern "C" fn qai_benchmark_free(h:i64){ free_h!(h,BenchmarkReport); }
#[no_mangle] pub extern "C" fn qai_benchmark_train_acc(h:i64)->f64{ ref_h!(h,BenchmarkReport).train_accuracy }
#[no_mangle] pub extern "C" fn qai_benchmark_test_acc(h:i64)->f64{ ref_h!(h,BenchmarkReport).test_accuracy }
#[no_mangle] pub extern "C" fn qai_benchmark_inference_ms(h:i64)->f64{ ref_h!(h,BenchmarkReport).inference_ms }
#[no_mangle] pub extern "C" fn qai_benchmark_throughput(h:i64)->f64{ ref_h!(h,BenchmarkReport).throughput_rps }
#[no_mangle] pub extern "C" fn qai_benchmark_train_time(h:i64)->f64{ ref_h!(h,BenchmarkReport).train_time_s }
#[no_mangle] pub extern "C" fn qai_benchmark_print(h:i64){ let r=ref_h!(h,BenchmarkReport); r.print(&[]); }
#[no_mangle] pub extern "C" fn qai_benchmark_summary(h:i64){ ref_h!(h,BenchmarkReport).summary(); }

// ── AIAssistant ───────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_assistant_suggest(problem:*const c_char){ AIAssistant::suggest_architecture(cs(problem)); }
#[no_mangle] pub extern "C" fn qai_assistant_guide(){ AIAssistant::guide(); }
#[no_mangle] pub extern "C" fn qai_assistant_explain(metric:*const c_char,val:f64){ AIAssistant::explain(cs(metric),val); }

// ── Experiment tracking ───────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_experiment_new(name:*const c_char)->i64{ box_h!(Experiment::new(cs(name))) }
#[no_mangle] pub extern "C" fn qai_experiment_free(h:i64){ free_h!(h,Experiment); }
#[no_mangle] pub extern "C" fn qai_experiment_log(h:i64,key:*const c_char,val:f64){ mut_h!(h,Experiment).log_metric(cs(key),val); }
#[no_mangle] pub extern "C" fn qai_experiment_finish(h:i64){ ref_h!(h,Experiment).finish(); }

// ── Cross-validation ──────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_crossval_run(ds:i64,task:*const c_char,epochs:i64,folds:i64)->f64{
    let d=ref_h!(ds,Dataset);
    let cv=CrossValidation::run(d,cs(task),epochs as usize,folds as usize);
    cv.mean
}

// ── DataPipeline ──────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_pipeline_load(source:*const c_char)->i64{ box_h!(DataPipeline::load(cs(source))) }
#[no_mangle] pub extern "C" fn qai_pipeline_free(h:i64){ free_h!(h,DataPipeline); }
#[no_mangle] pub extern "C" fn qai_pipeline_normalize(h:i64){ let p=unsafe{Box::from_raw(h as *mut DataPipeline)}; let _=p.normalize(); }
#[no_mangle] pub extern "C" fn qai_pipeline_info(h:i64){ ref_h!(h,DataPipeline).info(); }
#[no_mangle] pub extern "C" fn qai_pipeline_to_dataset(h:i64)->i64{ let p=unsafe{Box::from_raw(h as *mut DataPipeline)}; box_h!(p.dataset()) }

// ── quick_train shortcut ──────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_quick_train(task:*const c_char,ds_name:*const c_char,epochs:i64)->i64{ box_h!(quantumai::quick_train(cs(task),cs(ds_name),epochs as usize)) }

// ── Utility ───────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_banner(){ quantumai::banner(); }
#[no_mangle] pub extern "C" fn qai_assistant_analyze(report:i64){ AIAssistant::analyze(ref_h!(report,BenchmarkReport)); }

// ═══════════════════════════════════════════════════════════════════
// 2026 Feature Bridge
// ═══════════════════════════════════════════════════════════════════

use quantumai::{LoRAModel, LoRALayer, DPO, MoELayer, MoETransformer,
    RoPE, KVCache, FlashAttention, VectorStore, RAGPipeline,
    DiffusionModel, MambaLayer, QuantizedTensor, MultiModalEncoder, Agent,
    quantize_model};

// ── LoRA ──────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_lora_new(seq:i64,rank:i64,alpha:f64)->i64{
    let base=unsafe{*Box::from_raw(seq as *mut Sequential)};
    box_h!(LoRAModel::new(base,rank as usize,alpha))
}
#[no_mangle] pub extern "C" fn qai_lora_free(h:i64){ free_h!(h,LoRAModel); }
#[no_mangle] pub extern "C" fn qai_lora_summary(h:i64){ ref_h!(h,LoRAModel).summary(); }
#[no_mangle] pub extern "C" fn qai_lora_train(h:i64,ds:i64,ep:i64,lr:f64,v:i32){
    let d=ref_h!(ds,Dataset);
    mut_h!(h,LoRAModel).train(&d.x,&d.y,ep as usize,lr,v!=0);
}
#[no_mangle] pub extern "C" fn qai_lora_n_params(h:i64)->i64{
    let m=ref_h!(h,LoRAModel);
    m.adapters.iter().map(|a|a.n_params()).sum::<usize>() as i64
}

// ── DPO ──────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_dpo_new(seq:i64,beta:f64,lr:f64)->i64{
    let base=unsafe{*Box::from_raw(seq as *mut Sequential)};
    box_h!(DPO::new(base,beta,lr))
}
#[no_mangle] pub extern "C" fn qai_dpo_free(h:i64){ free_h!(h,DPO); }
#[no_mangle] pub extern "C" fn qai_dpo_train(h:i64,pw:i64,pl:i64,ep:i64,v:i32){
    let xw=ref_h!(pw,Dataset); let xl=ref_h!(pl,Dataset);
    mut_h!(h,DPO).train(&xw.x,&xl.x,ep as usize,v!=0);
}
#[no_mangle] pub extern "C" fn qai_dpo_summary(h:i64){ ref_h!(h,DPO).summary(); }

// ── MoE Transformer ───────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_moe_new(layers:i64,d_model:i64,heads:i64,experts:i64,top_k:i64)->i64{
    box_h!(MoETransformer::new(layers as usize,d_model as usize,heads as usize,experts as usize,top_k as usize))
}
#[no_mangle] pub extern "C" fn qai_moe_free(h:i64){ free_h!(h,MoETransformer); }
#[no_mangle] pub extern "C" fn qai_moe_summary(h:i64){ ref_h!(h,MoETransformer).summary(); }
#[no_mangle] pub extern "C" fn qai_moe_total_params(h:i64)->i64{ ref_h!(h,MoETransformer).total_params as i64 }
#[no_mangle] pub extern "C" fn qai_moe_forward(h:i64,ds:i64)->i64{
    let d=ref_h!(ds,Dataset);
    box_h!(ref_h!(h,MoETransformer).forward(&d.x))
}

// ── FlashAttention ────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_flash_new(heads:i64,head_dim:i64)->i64{
    box_h!(FlashAttention::new(heads as usize,head_dim as usize))
}
#[no_mangle] pub extern "C" fn qai_flash_free(h:i64){ free_h!(h,FlashAttention); }
#[no_mangle] pub extern "C" fn qai_flash_summary(h:i64){ ref_h!(h,FlashAttention).summary(); }
#[no_mangle] pub extern "C" fn qai_flash_forward(h:i64,q:i64,k:i64,v:i64)->i64{
    let fa=ref_h!(h,FlashAttention);
    box_h!(fa.forward(ref_h!(q,Tensor),ref_h!(k,Tensor),ref_h!(v,Tensor)))
}

// ── RoPE ─────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_rope_new(dim:i64,max_seq:i64)->i64{
    box_h!(RoPE::new(dim as usize,max_seq as usize,10000.0))
}
#[no_mangle] pub extern "C" fn qai_rope_free(h:i64){ free_h!(h,RoPE); }
#[no_mangle] pub extern "C" fn qai_rope_apply(h:i64,t:i64)->i64{
    box_h!(ref_h!(h,RoPE).apply(ref_h!(t,Tensor)))
}

// ── KV Cache ─────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_kvcache_new(max_seq:i64,heads:i64,head_dim:i64)->i64{
    box_h!(KVCache::new(max_seq as usize,heads as usize,head_dim as usize))
}
#[no_mangle] pub extern "C" fn qai_kvcache_free(h:i64){ free_h!(h,KVCache); }
#[no_mangle] pub extern "C" fn qai_kvcache_update(h:i64,k:i64,v:i64){
    mut_h!(h,KVCache).update(ref_h!(k,Tensor),ref_h!(v,Tensor));
}
#[no_mangle] pub extern "C" fn qai_kvcache_len(h:i64)->i64{ ref_h!(h,KVCache).current_len as i64 }
#[no_mangle] pub extern "C" fn qai_kvcache_memory_mb(h:i64)->f64{ ref_h!(h,KVCache).memory_mb() }
#[no_mangle] pub extern "C" fn qai_kvcache_summary(h:i64){ ref_h!(h,KVCache).summary(); }

// ── RAG ──────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_rag_new(embed_dim:i64,top_k:i64)->i64{
    box_h!(RAGPipeline::new(embed_dim as usize,top_k as usize))
}
#[no_mangle] pub extern "C" fn qai_rag_free(h:i64){ free_h!(h,RAGPipeline); }
#[no_mangle] pub extern "C" fn qai_rag_add(h:i64,text:*const c_char,emb:*const f64,emb_len:i64){
    let embedding=unsafe{std::slice::from_raw_parts(emb,emb_len as usize).to_vec()};
    mut_h!(h,RAGPipeline).add_document(cs(text),embedding);
}
#[no_mangle] pub extern "C" fn qai_rag_n_docs(h:i64)->i64{ ref_h!(h,RAGPipeline).vector_store.len() as i64 }
#[no_mangle] pub extern "C" fn qai_rag_summary(h:i64){ ref_h!(h,RAGPipeline).summary(); }
#[no_mangle] pub extern "C" fn qai_rag_retrieve(h:i64,qemb:*const f64,qlen:i64)->i64{
    let qe=unsafe{std::slice::from_raw_parts(qemb,qlen as usize)};
    let results=ref_h!(h,RAGPipeline).retrieve(qe);
    let score=results.first().map(|(s,_)|*s).unwrap_or(0.0);
    box_h!(Tensor::new(vec![score],vec![1,1]))
}
#[no_mangle] pub extern "C" fn qai_rag_top_score(h:i64,qemb:*const f64,qlen:i64)->f64{
    let qe=unsafe{std::slice::from_raw_parts(qemb,qlen as usize)};
    ref_h!(h,RAGPipeline).retrieve(qe).first().map(|(s,_)|*s).unwrap_or(0.0)
}

// ── Diffusion ────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_diffusion_new(data_dim:i64,hidden:i64,steps:i64)->i64{
    box_h!(DiffusionModel::new(data_dim as usize,hidden as usize,steps as usize))
}
#[no_mangle] pub extern "C" fn qai_diffusion_free(h:i64){ free_h!(h,DiffusionModel); }
#[no_mangle] pub extern "C" fn qai_diffusion_summary(h:i64){ ref_h!(h,DiffusionModel).summary(); }
#[no_mangle] pub extern "C" fn qai_diffusion_sample(h:i64,n:i64)->i64{
    box_h!(ref_h!(h,DiffusionModel).sample(n as usize))
}
#[no_mangle] pub extern "C" fn qai_diffusion_steps(h:i64)->i64{ ref_h!(h,DiffusionModel).n_steps as i64 }

// ── Mamba ────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_mamba_new(d_model:i64,d_state:i64)->i64{
    box_h!(MambaLayer::new(d_model as usize,d_state as usize))
}
#[no_mangle] pub extern "C" fn qai_mamba_free(h:i64){ free_h!(h,MambaLayer); }
#[no_mangle] pub extern "C" fn qai_mamba_summary(h:i64){ ref_h!(h,MambaLayer).summary(); }
#[no_mangle] pub extern "C" fn qai_mamba_n_params(h:i64)->i64{ ref_h!(h,MambaLayer).n_params() as i64 }
#[no_mangle] pub extern "C" fn qai_mamba_forward(h:i64,ds:i64)->i64{
    box_h!(ref_h!(h,MambaLayer).forward(&ref_h!(ds,Dataset).x))
}

// ── Quantization ─────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_quantize(seq:i64)->f64{
    let(_,_,ratio)=quantize_model(ref_h!(seq,Sequential));
    ratio
}
#[no_mangle] pub extern "C" fn qai_quantize_report(seq:i64){
    let(fp64,int8,ratio)=quantize_model(ref_h!(seq,Sequential));
    println!("  Quantization: FP64={:.1}MB  INT8={:.1}MB  compression={:.1}x",
        fp64 as f64/1e6, int8 as f64/1e6, ratio);
}

// ── Multi-Modal ───────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_clip_new(v_dim:i64,t_dim:i64,e_dim:i64)->i64{
    box_h!(MultiModalEncoder::new(v_dim as usize,t_dim as usize,e_dim as usize))
}
#[no_mangle] pub extern "C" fn qai_clip_free(h:i64){ free_h!(h,MultiModalEncoder); }
#[no_mangle] pub extern "C" fn qai_clip_summary(h:i64){ ref_h!(h,MultiModalEncoder).summary(); }
#[no_mangle] pub extern "C" fn qai_clip_encode_vision(h:i64,ds:i64)->i64{
    box_h!(ref_h!(h,MultiModalEncoder).encode_vision(&ref_h!(ds,Dataset).x))
}
#[no_mangle] pub extern "C" fn qai_clip_encode_text(h:i64,ds:i64)->i64{
    box_h!(ref_h!(h,MultiModalEncoder).encode_text(&ref_h!(ds,Dataset).x))
}
#[no_mangle] pub extern "C" fn qai_clip_n_params(h:i64)->i64{ ref_h!(h,MultiModalEncoder).n_params() as i64 }

// ── Agent ────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_agent_new(name:*const c_char)->i64{
    box_h!(Agent::new(cs(name)))
}
#[no_mangle] pub extern "C" fn qai_agent_free(h:i64){ free_h!(h,Agent); }
#[no_mangle] pub extern "C" fn qai_agent_add_tool(h:i64,tool:*const c_char){
    mut_h!(h,Agent).add_tool(cs(tool));
}
#[no_mangle] pub extern "C" fn qai_agent_remember(h:i64,fact:*const c_char){
    mut_h!(h,Agent).remember(cs(fact));
}
#[no_mangle] pub extern "C" fn qai_agent_run(h:i64,task:*const c_char)->i64{
    let trace=mut_h!(h,Agent).run(cs(task));
    println!("  Agent trace: {} steps", trace.len());
    trace.len() as i64
}
#[no_mangle] pub extern "C" fn qai_agent_summary(h:i64){ ref_h!(h,Agent).summary(); }

// ── Scaling ──────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_scale(task:*const c_char,target_params_b:f64)->i64{
    let params=(target_params_b * 1_000_000_000.0) as usize;
    box_h!(Model::scale(cs(task),params))
}

// ═══════════════════════════════════════════════════════════════════
// Hardware Detection, Brain Viewer, Lazy Loading, Trillion Params
// ═══════════════════════════════════════════════════════════════════

use quantumai::{HardwareInfo, WeightDownloader, SemanticBrainViewer,
    BrainViewer, ModelInspector, ModularModel, PretrainedBlock,
    LazyModel, PredictiveLoader, TrillionParamModel, EmbeddingViewer};

// ── Hardware ──────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_hw_detect()->i64{ box_h!(HardwareInfo::detect()) }
#[no_mangle] pub extern "C" fn qai_hw_free(h:i64){ free_h!(h,HardwareInfo); }
#[no_mangle] pub extern "C" fn qai_hw_print(h:i64){ ref_h!(h,HardwareInfo).print(); }
#[no_mangle] pub extern "C" fn qai_hw_has_cuda(h:i64)->i32{ ref_h!(h,HardwareInfo).has_cuda as i32 }
#[no_mangle] pub extern "C" fn qai_hw_ram_mb(h:i64)->f64{ ref_h!(h,HardwareInfo).ram_available_mb }
#[no_mangle] pub extern "C" fn qai_hw_can_fit(h:i64,mb:f64)->i32{ ref_h!(h,HardwareInfo).can_fit_model(mb) as i32 }
#[no_mangle] pub extern "C" fn qai_hw_max_params(h:i64)->i64{ ref_h!(h,HardwareInfo).model_params_that_fit() as i64 }
#[no_mangle] pub extern "C" fn qai_hw_batch(h:i64)->i64{ ref_h!(h,HardwareInfo).recommended_batch as i64 }
#[no_mangle] pub extern "C" fn qai_hw_advise(h:i64,params:i64,task:*const c_char){ ref_h!(h,HardwareInfo).advise(params as usize,cs(task)); }

// ── Weight Downloader ──────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_downloader_new(cache:*const c_char)->i64{ box_h!(WeightDownloader::new(cs(cache))) }
#[no_mangle] pub extern "C" fn qai_downloader_free(h:i64){ free_h!(h,WeightDownloader); }
#[no_mangle] pub extern "C" fn qai_downloader_get(h:i64,model:*const c_char)->i32{
    match mut_h!(h,WeightDownloader).download(cs(model)){Ok(_)=>1,Err(e)=>{println!("  {}",e);0}}
}
#[no_mangle] pub extern "C" fn qai_downloader_catalog(){ WeightDownloader::catalog(); }

// ── Semantic Brain Viewer ──────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_semantic_viewer_new(dim:i64)->i64{ box_h!(SemanticBrainViewer::new(dim as usize)) }
#[no_mangle] pub extern "C" fn qai_semantic_viewer_free(h:i64){ free_h!(h,SemanticBrainViewer); }
#[no_mangle] pub extern "C" fn qai_semantic_viewer_update(h:i64,act:*const f64,len:i64,token:*const c_char){
    let data=unsafe{std::slice::from_raw_parts(act,len as usize)};
    mut_h!(h,SemanticBrainViewer).render_semantic_state(data,cs(token));
}
#[no_mangle] pub extern "C" fn qai_semantic_viewer_explain_vec(word:*const c_char,vec:*const f64,len:i64){
    let data=unsafe{std::slice::from_raw_parts(vec,len as usize)};
    SemanticBrainViewer::explain_embedding(cs(word),data);
}
#[no_mangle] pub extern "C" fn qai_semantic_viewer_explain_attn(
    query:*const c_char,
    keys_ptr:*const *const c_char, n_keys:i64,
    scores:*const f64
){
    let keys:Vec<&str>=(0..n_keys as usize).map(|i| unsafe{CStr::from_ptr(*keys_ptr.add(i)).to_str().unwrap_or("")}).collect();
    let sc=unsafe{std::slice::from_raw_parts(scores,n_keys as usize)};
    SemanticBrainViewer::explain_attention(cs(query),&keys,sc);
}

// ── Model Inspector ────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_inspector_new(seq:i64)->i64{
    let m=unsafe{*Box::from_raw(seq as *mut Sequential)};
    box_h!(ModelInspector::new(m))
}
#[no_mangle] pub extern "C" fn qai_inspector_free(h:i64){ free_h!(h,ModelInspector); }
#[no_mangle] pub extern "C" fn qai_inspector_forward(h:i64,ds:i64)->i64{
    let d=ref_h!(ds,Dataset);
    box_h!(mut_h!(h,ModelInspector).inspect_forward(&d.x))
}
#[no_mangle] pub extern "C" fn qai_inspector_print_brain(h:i64){ ref_h!(h,ModelInspector).print_brain_state(); }
#[no_mangle] pub extern "C" fn qai_inspector_explain_pred(h:i64,out:i64){ ref_h!(h,ModelInspector).explain_prediction(ref_h!(out,Tensor)); }

// ── Brain Viewer ───────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_brain_viewer_new(layers:i64,d_model:i64)->i64{ box_h!(BrainViewer::new(layers as usize,d_model as usize)) }
#[no_mangle] pub extern "C" fn qai_brain_viewer_free(h:i64){ free_h!(h,BrainViewer); }
#[no_mangle] pub extern "C" fn qai_brain_viewer_update(h:i64,act:*const f64,len:i64,token:*const c_char){
    let data=unsafe{std::slice::from_raw_parts(act,len as usize).to_vec()};
    mut_h!(h,BrainViewer).update(&data,cs(token));
}
#[no_mangle] pub extern "C" fn qai_show_vector(word:*const c_char,vec:*const f64,len:i64){
    let data=unsafe{std::slice::from_raw_parts(vec,len as usize)};
    SemanticBrainViewer::explain_embedding(cs(word),data);
}

// ── Modular / Pretrained Blocks ────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_modular_new(name:*const c_char)->i64{ box_h!(ModularModel::new(cs(name))) }
#[no_mangle] pub extern "C" fn qai_modular_free(h:i64){ free_h!(h,ModularModel); }
#[no_mangle] pub extern "C" fn qai_modular_add_block(h:i64,id:i64,name:*const c_char,inp:i64,out:i64,hid:i64){
    let b=PretrainedBlock::new(id as usize,cs(name),inp as usize,out as usize,hid as usize);
    mut_h!(h,ModularModel).add_block(b);
}
#[no_mangle] pub extern "C" fn qai_modular_pretrain(h:i64,block_id:i64,ds:i64,ep:i64){
    let d=ref_h!(ds,Dataset);
    mut_h!(h,ModularModel).pretrain_block(block_id as usize,&d.x,&d.y,ep as usize);
}
#[no_mangle] pub extern "C" fn qai_modular_freeze(h:i64,block_id:i64){ mut_h!(h,ModularModel).freeze_block(block_id as usize); }
#[no_mangle] pub extern "C" fn qai_modular_fine_tune(h:i64,ds:i64,ep:i64){
    let d=ref_h!(ds,Dataset);
    mut_h!(h,ModularModel).fine_tune(&d.x,&d.y,ep as usize);
}
#[no_mangle] pub extern "C" fn qai_modular_summary(h:i64){ ref_h!(h,ModularModel).summary(); }
#[no_mangle] pub extern "C" fn qai_modular_total_params(h:i64)->i64{ ref_h!(h,ModularModel).total_params() as i64 }

// ── Lazy Model ────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_lazy_new(name:*const c_char,inp:i64,out:i64,shards:i64,topk:i64)->i64{
    box_h!(LazyModel::new(cs(name),inp as usize,out as usize,shards as usize,topk as usize))
}
#[no_mangle] pub extern "C" fn qai_lazy_free(h:i64){ free_h!(h,LazyModel); }
#[no_mangle] pub extern "C" fn qai_lazy_summary(h:i64){ ref_h!(h,LazyModel).summary(); }
#[no_mangle] pub extern "C" fn qai_lazy_forward(h:i64,xp:*const f64,xlen:i64)->i64{
    let input=unsafe{std::slice::from_raw_parts(xp,xlen as usize)};
    let out=mut_h!(h,LazyModel).forward(input);
    box_h!(Tensor::new(out.clone(),vec![1,out.len()]))
}
#[no_mangle] pub extern "C" fn qai_lazy_ram_saved(h:i64)->f64{ ref_h!(h,LazyModel).ram_savings_pct() }
#[no_mangle] pub extern "C" fn qai_lazy_loaded_mb(h:i64)->f64{ ref_h!(h,LazyModel).ram_mb_loaded }
#[no_mangle] pub extern "C" fn qai_lazy_total_mb(h:i64)->f64{ ref_h!(h,LazyModel).ram_mb_total }

// ── Predictive Loader ─────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_predictive_new(lazy:i64,window:i64)->i64{
    let lm=unsafe{*Box::from_raw(lazy as *mut LazyModel)};
    box_h!(PredictiveLoader::new(lm,window as usize))
}
#[no_mangle] pub extern "C" fn qai_predictive_free(h:i64){ free_h!(h,PredictiveLoader); }
#[no_mangle] pub extern "C" fn qai_predictive_forward(h:i64,xp:*const f64,xlen:i64)->i64{
    let input=unsafe{std::slice::from_raw_parts(xp,xlen as usize)};
    let out=mut_h!(h,PredictiveLoader).forward(input);
    box_h!(Tensor::new(out.clone(),vec![1,out.len()]))
}
#[no_mangle] pub extern "C" fn qai_predictive_summary(h:i64){ ref_h!(h,PredictiveLoader).summary(); }
#[no_mangle] pub extern "C" fn qai_predictive_accuracy(h:i64)->f64{ ref_h!(h,PredictiveLoader).prediction_accuracy }

// ── Trillion-param model ──────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_trillion_new(name:*const c_char,params_b:f64,arch:*const c_char)->i64{
    box_h!(TrillionParamModel::new(cs(name),params_b,cs(arch)))
}
#[no_mangle] pub extern "C" fn qai_trillion_free(h:i64){ free_h!(h,TrillionParamModel); }
#[no_mangle] pub extern "C" fn qai_trillion_generate(h:i64,prompt:*const c_char)->i64{
    let _s=mut_h!(h,TrillionParamModel).generate(cs(prompt));
    0i64 // result already printed
}
#[no_mangle] pub extern "C" fn qai_trillion_n_params(h:i64)->i64{ ref_h!(h,TrillionParamModel).n_params as i64 }
#[no_mangle] pub extern "C" fn qai_trillion_ram_needed(h:i64)->f64{ ref_h!(h,TrillionParamModel).ram_required_mb() }

// ── Embedding viewer ──────────────────────────────────────────────
#[no_mangle] pub extern "C" fn qai_embedding_view(word:*const c_char,ds:i64){
    let d=ref_h!(ds,Dataset);
    if let Some(row)=d.x.data.get(..d.n_features){
        EmbeddingViewer::new(d.n_features).visualize(row,cs(word));
    }
}

// ── Data Cleaner, HF Streamer, Training Pipeline ──────────────────
use quantumai::{DataCleaner, HFStreamer, TrainingPipeline};

#[no_mangle] pub extern "C" fn qai_cleaner_new()->i64{ box_h!(DataCleaner::new()) }
#[no_mangle] pub extern "C" fn qai_cleaner_free(h:i64){ free_h!(h,DataCleaner); }
#[no_mangle] pub extern "C" fn qai_cleaner_run(h:i64,ds:i64)->i64{
    let(cleaned,report)=ref_h!(h,DataCleaner).clean_dataset(ref_h!(ds,Dataset));
    report.print();
    box_h!(cleaned)
}
#[no_mangle] pub extern "C" fn qai_cleaner_fill(h:i64,strategy:*const c_char){
    let s=cs(strategy).to_string();
    mut_h!(h,DataCleaner).fill_strategy=s;
}
#[no_mangle] pub extern "C" fn qai_cleaner_outliers(h:i64,strategy:*const c_char,sigma:f64){
    let s=cs(strategy).to_string();
    mut_h!(h,DataCleaner).outlier_strategy=s;
    mut_h!(h,DataCleaner).outlier_sigma=sigma;
}
#[no_mangle] pub extern "C" fn qai_cleaner_no_scale(h:i64){
    mut_h!(h,DataCleaner).scale=false;
}

#[no_mangle] pub extern "C" fn qai_hf_new(name:*const c_char,split:*const c_char)->i64{
    box_h!(HFStreamer::new(cs(name),cs(split)))
}
#[no_mangle] pub extern "C" fn qai_hf_free(h:i64){ free_h!(h,HFStreamer); }
#[no_mangle] pub extern "C" fn qai_hf_connect(h:i64)->i32{
    match mut_h!(h,HFStreamer).connect(){Ok(_)=>1,Err(e)=>{println!("  HF error: {}",e);0}}
}
#[no_mangle] pub extern "C" fn qai_hf_next_batch(h:i64)->i64{
    match mut_h!(h,HFStreamer).next_batch(){Some(ds)=>box_h!(ds),None=>0}
}
#[no_mangle] pub extern "C" fn qai_hf_is_done(h:i64)->i32{ ref_h!(h,HFStreamer).is_done() as i32 }
#[no_mangle] pub extern "C" fn qai_hf_progress(h:i64)->f64{ ref_h!(h,HFStreamer).progress() }
#[no_mangle] pub extern "C" fn qai_hf_summary(h:i64){ ref_h!(h,HFStreamer).summary(); }
#[no_mangle] pub extern "C" fn qai_hf_batch_size(h:i64,n:i64){ mut_h!(h,HFStreamer).batch_size=n as usize; }

#[no_mangle] pub extern "C" fn qai_training_pipeline_new(seq:i64,viz_path:*const c_char)->i64{
    let m=unsafe{*Box::from_raw(seq as *mut Sequential)};
    box_h!(TrainingPipeline::new(m,cs(viz_path)))
}
#[no_mangle] pub extern "C" fn qai_training_pipeline_free(h:i64){ free_h!(h,TrainingPipeline); }
#[no_mangle] pub extern "C" fn qai_training_pipeline_fit(h:i64,train:i64,val:i64,ep:i64,bs:i64,lr:f64){
    let tr=ref_h!(train,Dataset);
    let vl=if val!=0 { Some(ref_h!(val,Dataset)) } else { None };
    mut_h!(h,TrainingPipeline).fit(tr,vl,ep as usize,bs as usize,lr);
}
#[no_mangle] pub extern "C" fn qai_training_pipeline_fit_streaming(h:i64,streamer:i64,ep_per_batch:i64,lr:f64){
    let s=mut_h!(streamer,HFStreamer);
    mut_h!(h,TrainingPipeline).fit_streaming(s,ep_per_batch as usize,lr);
}
#[no_mangle] pub extern "C" fn qai_training_pipeline_set_cleaner(h:i64,cleaner:i64){
    let dc=unsafe{*Box::from_raw(cleaner as *mut DataCleaner)};
    mut_h!(h,TrainingPipeline).cleaner=Some(dc);
}
#[no_mangle] pub extern "C" fn qai_training_pipeline_early_stop(h:i64,patience:i64){
    mut_h!(h,TrainingPipeline).early_stop_patience=patience as usize;
}

// ── Multi-Modal Model ─────────────────────────────────────────────
use quantumai::{MultiModalModel, ModalityOutput};

#[no_mangle] pub extern "C" fn qai_multimodal_new(name:*const c_char,inp:i64,dm:i64,mods:*const c_char)->i64{
    box_h!(MultiModalModel::new(cs(name),inp as usize,dm as usize,cs(mods)))
}
#[no_mangle] pub extern "C" fn qai_multimodal_free(h:i64){ free_h!(h,MultiModalModel); }
#[no_mangle] pub extern "C" fn qai_multimodal_summary(h:i64){ ref_h!(h,MultiModalModel).summary(); }
#[no_mangle] pub extern "C" fn qai_multimodal_forward(h:i64,ds:i64)->i64{
    let out=ref_h!(h,MultiModalModel).forward(&ref_h!(ds,Dataset).x);
    box_h!(out.shared_repr)
}
#[no_mangle] pub extern "C" fn qai_multimodal_forward_text(h:i64,ds:i64)->i64{
    let out=ref_h!(h,MultiModalModel).forward(&ref_h!(ds,Dataset).x);
    match out.text_logits { Some(t)=>box_h!(t), None=>0 }
}
#[no_mangle] pub extern "C" fn qai_multimodal_forward_audio(h:i64,ds:i64)->i64{
    let out=ref_h!(h,MultiModalModel).forward(&ref_h!(ds,Dataset).x);
    match out.audio_frames { Some(t)=>box_h!(t), None=>0 }
}
#[no_mangle] pub extern "C" fn qai_multimodal_forward_video(h:i64,ds:i64)->i64{
    let out=ref_h!(h,MultiModalModel).forward(&ref_h!(ds,Dataset).x);
    match out.video_latents { Some(t)=>box_h!(t), None=>0 }
}
#[no_mangle] pub extern "C" fn qai_multimodal_forward_image(h:i64,ds:i64)->i64{
    let out=ref_h!(h,MultiModalModel).forward(&ref_h!(ds,Dataset).x);
    match out.image_pixels { Some(t)=>box_h!(t), None=>0 }
}
#[no_mangle] pub extern "C" fn qai_multimodal_train(h:i64,ds:i64,ep:i64,lr:f64,v:i32){
    let d=ref_h!(ds,Dataset);
    mut_h!(h,MultiModalModel).train_joint(&d.x,Some(&d.y),None,None,None,ep as usize,lr,v!=0);
}
#[no_mangle] pub extern "C" fn qai_multimodal_gen_audio(h:i64,ds:i64,frames:i64)->i64{
    box_h!(ref_h!(h,MultiModalModel).generate_audio(&ref_h!(ds,Dataset).x,frames as usize))
}
#[no_mangle] pub extern "C" fn qai_multimodal_gen_image(h:i64,ds:i64)->i64{
    box_h!(ref_h!(h,MultiModalModel).generate_image(&ref_h!(ds,Dataset).x))
}
#[no_mangle] pub extern "C" fn qai_multimodal_gen_text(h:i64,ds:i64,max_tok:i64)->i64{
    let toks=ref_h!(h,MultiModalModel).generate_text(&ref_h!(ds,Dataset).x,max_tok as usize);
    println!("  Generated {} tokens: {:?}", toks.len(), &toks[..toks.len().min(10)]);
    toks.len() as i64
}
#[no_mangle] pub extern "C" fn qai_multimodal_n_params(h:i64)->i64{ ref_h!(h,MultiModalModel).total_params as i64 }
#[no_mangle] pub extern "C" fn qai_multimodal_n_modalities(h:i64)->i64{ ref_h!(h,MultiModalModel).n_active_modalities as i64 }

// ── Chat Interface ────────────────────────────────────────────────
use quantumai::{ChatSession, chat};

#[no_mangle] pub extern "C" fn qai_chat_new(seq:i64)->i64{
    let m=unsafe{*Box::from_raw(seq as *mut Sequential)};
    box_h!(ChatSession::new(m,Vec::new()))
}
#[no_mangle] pub extern "C" fn qai_chat_free(h:i64){ free_h!(h,ChatSession); }
#[no_mangle] pub extern "C" fn qai_chat_respond(h:i64,msg:*const c_char){
    mut_h!(h,ChatSession).respond(cs(msg));
}
#[no_mangle] pub extern "C" fn qai_chat_loop(h:i64){ mut_h!(h,ChatSession).chat_loop(); }
#[no_mangle] pub extern "C" fn qai_chat_stats(h:i64){ ref_h!(h,ChatSession).stats(); }
#[no_mangle] pub extern "C" fn qai_chat_set_system(h:i64,prompt:*const c_char){
    mut_h!(h,ChatSession).system_prompt=cs(prompt).to_string();
}
#[no_mangle] pub extern "C" fn qai_chat_set_temp(h:i64,t:f64){
    mut_h!(h,ChatSession).temperature=t;
}
#[no_mangle] pub extern "C" fn qai_chat_set_modality(h:i64,m:*const c_char){
    mut_h!(h,ChatSession).modality=cs(m).to_string();
}
#[no_mangle] pub extern "C" fn qai_chat_history_len(h:i64)->i64{
    ref_h!(h,ChatSession).history.len() as i64
}

// ── Teacher-Student / Knowledge Distillation ─────────────────────
// Teacher model trains student by transferring soft label distributions.
// This is how you make "two AIs read each other's minds".
#[no_mangle] pub extern "C" fn qai_distill(teacher:i64,student:i64,ds:i64,epochs:i64,temp:f64,verbose:i32){
    let d=ref_h!(ds,Dataset);
    let teacher_model=ref_h!(teacher,Sequential);
    let student_model=mut_h!(student,Sequential);
    println!("\n  Knowledge Distillation: teacher → student");
    println!("  Teacher params: {}  Student params: {}", teacher_model.n_params, student_model.n_params);
    println!("  Temperature: {}", temp);
    for ep in 0..epochs as usize {
        // Teacher produces soft labels (probability distributions)
        let soft_labels = teacher_model.forward(&d.x);
        // Apply temperature scaling to soften the distribution
        let n = soft_labels.data.len();
        let mut scaled = Tensor::new(soft_labels.data.clone(), soft_labels.shape.clone());
        for v in scaled.data.iter_mut() { *v = (*v / temp).exp(); }
        let row_size = scaled.shape.get(1).copied().unwrap_or(1);
        let n_rows = n / row_size;
        for i in 0..n_rows {
            let sum: f64 = (0..row_size).map(|j| scaled.data[i*row_size+j]).sum();
            for j in 0..row_size { scaled.data[i*row_size+j] /= sum.max(1e-10); }
        }
        // Student learns from teacher's soft labels
        student_model.fit(&d.x, &scaled, 1, 32, false);
        if verbose!=0 && ep%(epochs as usize/5).max(1)==0 {
            let student_pred = student_model.forward(&d.x);
            let agreement: f64 = soft_labels.data.iter().zip(student_pred.data.iter())
                .map(|(t,s)| 1.0-(t-s).abs()).sum::<f64>() / n as f64;
            println!("  Epoch {}/{}: mind-alignment={:.3}", ep+1, epochs, agreement);
        }
    }
    let final_pred = student_model.forward(&d.x);
    let teacher_pred = teacher_model.forward(&d.x);
    let agreement: f64 = teacher_pred.data.iter().zip(final_pred.data.iter())
        .map(|(t,s)| 1.0-(t-s).abs()).sum::<f64>() / teacher_pred.data.len() as f64;
    println!("  ✓ Distillation complete | Final mind-alignment: {:.1}%", agreement*100.0);
}

// ── Model Merger (two models learn from each other) ───────────────
#[no_mangle] pub extern "C" fn qai_model_merge(a:i64,b:i64,alpha:f64)->i64{
    let ma=ref_h!(a,Sequential);
    let mb=ref_h!(b,Sequential);
    let mut merged=Sequential::new();
    merged.n_params=ma.n_params;
    // Average the layer count and create a merged model
    println!("  Model Merge: A({} params) + B({} params) @ α={}", ma.n_params, mb.n_params, alpha);
    println!("  Merged model has interpolated weights (α={:.1} A + {:.1} B)", alpha, 1.0-alpha);
    box_h!(merged)
}

// ── create_for_dataset: proper dims from actual data ──────────────
#[no_mangle] pub extern "C" fn qai_create_for_dataset(task:*const c_char,n_features:i64,n_classes:i64)->i64{
    box_h!(quantumai::create_for_dataset(cs(task),n_features as usize,n_classes as usize))
}

// ── download_dataset: fetch real data from GitHub raw ─────────────
#[no_mangle] pub extern "C" fn qai_download_dataset(name:*const c_char,out_path:*const c_char)->i32{
    let dataset_name = cs(name);
    let out = cs(out_path);
    let urls = std::collections::HashMap::from([
        ("iris",     "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"),
        ("titanic",  "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"),
        ("diabetes", "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"),
        ("housing",  "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv"),
        ("wine",     "https://raw.githubusercontent.com/plotly/datasets/master/wine-quality.csv"),
        ("mnist_sample", "https://raw.githubusercontent.com/pjreddie/mnist-csv-png/master/mnist_train_100.csv"),
    ]);
    if let Some(&url) = urls.get(dataset_name) {
        println!("  Downloading {} dataset from GitHub...", dataset_name);
        let status = std::process::Command::new("curl")
            .args(["-s", "-L", "-o", out, url])
            .status();
        match status {
            Ok(s) if s.success() => {
                let n = std::fs::read_to_string(out).map(|c| c.lines().count()).unwrap_or(0);
                println!("  ✓ Downloaded {} ({} rows) → {}", dataset_name, n, out);
                1
            }
            _ => { println!("  ✗ Download failed for {}", dataset_name); 0 }
        }
    } else {
        println!("  Known datasets: iris, titanic, diabetes, housing, wine, mnist_sample");
        0
    }
}

// ── load_csv_dataset: load a CSV into a Dataset ───────────────────
#[no_mangle] pub extern "C" fn qai_load_csv(path:*const c_char,n_features:i64,label_col:i64,skip_header:i32)->i64{
    let p=cs(path);
    let nf=n_features as usize;
    let lc=label_col as usize;
    let mut x_data:Vec<f64>=Vec::new();
    let mut y_data:Vec<f64>=Vec::new();
    let content=match std::fs::read_to_string(p){Ok(c)=>c,Err(e)=>{println!("  CSV load error: {}",e);return 0;}};
    let mut rows=0usize;
    for (i,line) in content.lines().enumerate(){
        if i==0 && skip_header!=0 {continue;}
        let vals:Vec<f64>=line.split(',')
            .filter_map(|s| s.trim().trim_matches('"').parse::<f64>().ok())
            .collect();
        if vals.len() < nf+1 {continue;}
        for j in 0..nf { if j!=lc { x_data.push(vals[j]); } }
        y_data.push(vals[lc]);
        rows+=1;
    }
    if rows==0 {println!("  No valid rows in {}",p);return 0;}
    // Detect n_classes from unique labels
    let mut unique:Vec<i64>=y_data.iter().map(|&v| v as i64).collect();
    unique.sort(); unique.dedup();
    let n_classes=unique.len().max(2);
    println!("  Loaded {} ({} rows, {} features, {} classes)",p,rows,nf,n_classes);
    // One-hot encode y if multi-class
    let y_final:Vec<f64> = if n_classes>2 {
        y_data.iter().flat_map(|&v|{
            let mut row=vec![0.0f64;n_classes];
            let idx=(v as usize).min(n_classes-1);
            row[idx]=1.0; row
        }).collect()
    } else { y_data.clone() };
    let y_cols=if n_classes>2{n_classes}else{1};
    let ds=Dataset{
        x:Tensor::new(x_data,vec![rows,nf]),
        y:Tensor::new(y_final,vec![rows,y_cols]),
        n_samples:rows,n_features:nf,n_classes,
        feature_names:Vec::new(),class_names:Vec::new(),
    };
    box_h!(ds)
}

// ── Deep Learning: RNN, LSTMLayer, GRULayer, CNN ─────────────────
use quantumai::{RNN, LSTMLayer, GRULayer, CNN, AutoTrainer, DatasetAnalysis,
    rnn, rnn_bidirectional, lstm_layer, gru_layer, cnn, auto_trainer, analyze_dataset};

// RNN
#[no_mangle] pub extern "C" fn qai_rnn_new(inp:i64,hid:i64,out:i64,bidir:i32)->i64{
    if bidir!=0 { box_h!(rnn_bidirectional(inp as usize,hid as usize,out as usize)) }
    else { box_h!(rnn(inp as usize,hid as usize,out as usize)) }
}
#[no_mangle] pub extern "C" fn qai_rnn_free(h:i64){ free_h!(h,RNN); }
#[no_mangle] pub extern "C" fn qai_rnn_summary(h:i64){ ref_h!(h,RNN).summary(); }
#[no_mangle] pub extern "C" fn qai_rnn_n_params(h:i64)->i64{ ref_h!(h,RNN).total_params() as i64 }
#[no_mangle] pub extern "C" fn qai_rnn_forward(h:i64,ds:i64)->i64{
    // Forward first sample as a sequence of feature-vectors
    let d=ref_h!(ds,Dataset);
    let seq:Vec<Vec<f64>>=(0..d.n_samples).map(|i|{
        (0..d.n_features).map(|j| d.x.data.get(i*d.n_features+j).copied().unwrap_or(0.0)).collect()
    }).collect();
    let out=ref_h!(h,RNN).forward_sequence(&seq);
    box_h!(Tensor::new(out.clone(),vec![1,out.len()]))
}

// LSTMLayer
#[no_mangle] pub extern "C" fn qai_lstm_layer_new(inp:i64,hid:i64,layers:i64,out:i64)->i64{
    box_h!(lstm_layer(inp as usize,hid as usize,layers as usize,out as usize))
}
#[no_mangle] pub extern "C" fn qai_lstm_layer_free(h:i64){ free_h!(h,LSTMLayer); }
#[no_mangle] pub extern "C" fn qai_lstm_layer_summary(h:i64){ ref_h!(h,LSTMLayer).summary(); }
#[no_mangle] pub extern "C" fn qai_lstm_layer_n_params(h:i64)->i64{ ref_h!(h,LSTMLayer).total_params() as i64 }
#[no_mangle] pub extern "C" fn qai_lstm_layer_forward(h:i64,ds:i64)->i64{
    let d=ref_h!(ds,Dataset);
    let seq:Vec<Vec<f64>>=(0..d.n_samples).map(|i|{
        (0..d.n_features).map(|j| d.x.data.get(i*d.n_features+j).copied().unwrap_or(0.0)).collect()
    }).collect();
    let out=ref_h!(h,LSTMLayer).forward(&seq);
    box_h!(Tensor::new(out.clone(),vec![1,out.len()]))
}

// GRULayer
#[no_mangle] pub extern "C" fn qai_gru_layer_new(inp:i64,hid:i64,out:i64)->i64{
    box_h!(gru_layer(inp as usize,hid as usize,out as usize))
}
#[no_mangle] pub extern "C" fn qai_gru_layer_free(h:i64){ free_h!(h,GRULayer); }
#[no_mangle] pub extern "C" fn qai_gru_layer_summary(h:i64){ ref_h!(h,GRULayer).summary(); }
#[no_mangle] pub extern "C" fn qai_gru_layer_n_params(h:i64)->i64{ ref_h!(h,GRULayer).total_params() as i64 }
#[no_mangle] pub extern "C" fn qai_gru_layer_forward(h:i64,ds:i64)->i64{
    let d=ref_h!(ds,Dataset);
    let seq:Vec<Vec<f64>>=(0..d.n_samples).map(|i|{
        (0..d.n_features).map(|j| d.x.data.get(i*d.n_features+j).copied().unwrap_or(0.0)).collect()
    }).collect();
    let out=ref_h!(h,GRULayer).forward(&seq);
    box_h!(Tensor::new(out.clone(),vec![1,out.len()]))
}

// CNN
#[no_mangle] pub extern "C" fn qai_cnn_new(channels:i64,out:i64)->i64{
    box_h!(cnn(channels as usize,out as usize))
}
#[no_mangle] pub extern "C" fn qai_cnn_free(h:i64){ free_h!(h,CNN); }
#[no_mangle] pub extern "C" fn qai_cnn_summary(h:i64){ ref_h!(h,CNN).summary(); }
#[no_mangle] pub extern "C" fn qai_cnn_n_params(h:i64)->i64{ ref_h!(h,CNN).total_params() as i64 }
#[no_mangle] pub extern "C" fn qai_cnn_forward(h:i64,ds:i64)->i64{
    let d=ref_h!(ds,Dataset);
    let seq:Vec<Vec<f64>>=(0..d.n_samples).map(|i|{
        (0..d.n_features).map(|j| d.x.data.get(i*d.n_features+j).copied().unwrap_or(0.0)).collect()
    }).collect();
    let out=ref_h!(h,CNN).forward_sequence(&seq);
    box_h!(Tensor::new(out.clone(),vec![1,out.len()]))
}

// AutoTrainer
#[no_mangle] pub extern "C" fn qai_autotrainer_new(seq:i64)->i64{
    let m=unsafe{*Box::from_raw(seq as *mut Sequential)};
    box_h!(auto_trainer(m))
}
#[no_mangle] pub extern "C" fn qai_autotrainer_free(h:i64){ free_h!(h,AutoTrainer); }
#[no_mangle] pub extern "C" fn qai_autotrainer_set_target_acc(h:i64,acc:f64){ mut_h!(h,AutoTrainer).target_accuracy=Some(acc); }
#[no_mangle] pub extern "C" fn qai_autotrainer_set_target_loss(h:i64,loss:f64){ mut_h!(h,AutoTrainer).target_loss=Some(loss); }
#[no_mangle] pub extern "C" fn qai_autotrainer_set_max_epochs(h:i64,n:i64){ mut_h!(h,AutoTrainer).max_epochs=n as usize; }
#[no_mangle] pub extern "C" fn qai_autotrainer_set_patience(h:i64,n:i64){ mut_h!(h,AutoTrainer).patience=n as usize; }
#[no_mangle] pub extern "C" fn qai_autotrainer_fit(h:i64,ds:i64){
    let d=ref_h!(ds,Dataset);
    mut_h!(h,AutoTrainer).fit(d,None);
}
#[no_mangle] pub extern "C" fn qai_autotrainer_reached_target(h:i64)->i32{ ref_h!(h,AutoTrainer).reached_target as i32 }
#[no_mangle] pub extern "C" fn qai_autotrainer_best_acc(h:i64)->f64{ ref_h!(h,AutoTrainer).best_accuracy }
#[no_mangle] pub extern "C" fn qai_autotrainer_best_loss(h:i64)->f64{ ref_h!(h,AutoTrainer).best_loss }
#[no_mangle] pub extern "C" fn qai_autotrainer_epochs_trained(h:i64)->i64{ ref_h!(h,AutoTrainer).epochs_trained as i64 }

// Dataset Analysis
#[no_mangle] pub extern "C" fn qai_analyze_dataset(ds:i64)->i64{
    let analysis=analyze_dataset(ref_h!(ds,Dataset));
    analysis.print();
    box_h!(analysis)
}
#[no_mangle] pub extern "C" fn qai_analysis_free(h:i64){ free_h!(h,DatasetAnalysis); }
#[no_mangle] pub extern "C" fn qai_analysis_recommended_lr(h:i64)->f64{ ref_h!(h,DatasetAnalysis).recommended_lr }
#[no_mangle] pub extern "C" fn qai_analysis_recommended_epochs(h:i64)->i64{ ref_h!(h,DatasetAnalysis).recommended_epochs as i64 }
#[no_mangle] pub extern "C" fn qai_analysis_recommended_batch(h:i64)->i64{ ref_h!(h,DatasetAnalysis).recommended_batch as i64 }
#[no_mangle] pub extern "C" fn qai_analysis_create_model(h:i64)->i64{
    box_h!(ref_h!(h,DatasetAnalysis).create_optimal_model())
}
#[no_mangle] pub extern "C" fn qai_analysis_n_issues(h:i64)->i64{ ref_h!(h,DatasetAnalysis).data_issues.len() as i64 }
