
import tensorflow as tf
import reader_1
import os

FLAGS = tf.flags.FLAGS

#tf.flags.DEFINE_string("model_dir", "", "The directory of the model file to evaluate.")

class SmallGenConfig(object):
  """Small config. for generation"""
  init_scale = 0.1
  learning_rate = 0.7
  max_grad_norm = 5
  num_layers = 2
  num_steps = 1 # this is the main difference
  hidden_size = 200
  max_epoch = 5
  max_max_epoch = 20
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 1
  vocab_size = 10000



def generate_text(train_path, model_path, num_sentences):
	
	gen_config = SmallGenConfig()
	
	checkpoint_file = tf.train.latest_checkpoint(model_path)

	with  tf.Graph().as_default(),tf.Session() as session: # tf.Graph().as_default(),


		#Method 1:

		#tf.initialize_all_variables().run()
		#saver = tf.train.Saver()
		#ckpt = tf.train.get_checkpoint_state(checkpoint_file)

		#if ckpt and ckpt.model_checkpoint_path:
			#saver.restore(session, ckpt.model_checkpoint_path)
		#print ("Session restored successfully!")


		# Method 2:

		print('loading checkpoint from from {}'.format(checkpoint_file))
		saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
		saver.restore(session, checkpoint_file)

		
		# Method 3:

		# Restore variables from disk.
		#saver = tf.train.Saver() 
		#print ("Restoring from : ", model_path)
		#saver.restore(session, model_path)
		#print("Model restored from file " + model_path)



		print("FLAGS.data_path -->",FLAGS.data_path)
		raw_data = reader.ptb_raw_data(FLAGS.data_path)
		train_data, valid_data, test_data, word_to_id, id_2_word = raw_data

		print("FLAGS.save_path -->",FLAGS.save_path)
		checkpoint_file = tf.train.latest_checkpoint(FLAGS.save_path)



		initializer = tf.random_uniform_initializer(-gen_config.init_scale,gen_config.init_scale)    

		with tf.variable_scope("model", reuse=None, initializer=initializer):
			train_input = PTBInput(config=gen_config, data=train_data, name="TrainInput")
			m = PTBModel(is_training=False, config=gen_config, input_=train_input)



		
		words = reader.get_vocab(train_path)

		state = m.initial_state.eval()
		x = 2 # the id for '<eos>' from the training set
		input = np.matrix([[x]])  # a 2D numpy matrix 

		text = ""
		count = 0
		while count < num_words:
			output_probs, state = session.run([m.output_probs, m.final_state],
																	 {m.input_data: input,
																		m.initial_state: state})
			x = sample(output_probs[0], 0.9)
			
			text += " " + words[x]
			count += 1

			# now feed this new word as input into the next iteration
			input = np.matrix([[x]])

		print(text)
	return


def main():


	train_path = "/Users/SeansMBP/Desktop/Cho/Project/data/tagged/train.txt"

	model_path = "/Users/SeansMBP/Desktop/Cho/Project/git stuff/grumpy/save_dir/tagged_hmlstm_small_drop_1213235534"

	num_words = 360

	generate_text(train_path, model_path, num_words)





if __name__ == '__main__':
	main()
