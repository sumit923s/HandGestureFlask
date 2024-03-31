import tensorflow as tf
import tf1.saved_model as sm

tf.compat.v1.enable_v2_behavior()

def convert_model(tf1_model_path, tf2_model_path):
    with tf.compat.v1.Session() as sess:
        tf1_model = sm.load(tf1_model_path, tags=['serve'])
        builder = tf.saved_model.builder.SavedModelBuilder(tf2_model_path)
        signature_def_map = {}
        for input_key in tf1_model.signature_def['serving_default'].inputs:
            input_key = input_key.name.replace(':0', '')
            signature_def_map[input_key] = tf1_model.signature_def['serving_default'].inputs[input_key]
        for output_key in tf1_model.signature_def['serving_default'].outputs:
            output_key = output_key.name.replace(':0', '')
            signature_def_map[output_key] = tf1_model.signature_def['serving_default'].outputs[output_key]
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], signature_def_map=signature_def_map)
        builder.save()

if __name__ == '__main__':
    tf1_model_path = 'mp_hand_gesture'
    tf2_model_path = 'mp_hand_gesture_tf2'
    convert_model(tf1_model_path, tf2_model_path)