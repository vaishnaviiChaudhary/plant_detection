import streamlit as st
import tensorflow as tf
import numpy as np
import pickle


st.set_page_config(page_title='CROPIFY', page_icon ='🌱')


# Tensorflow Model Prediction
def model_prediction(test_image):
    # Load the pickled model
    with open("trained_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element


# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition","Fertilizers", "Stastics"])

# Main Page
if(app_mode=="Home"):
    st.header("CROPIFY: Plant Disease Detection System")
    image_path = "homepage.png"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ## How It Works
    1. **Upload an Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.
    """)

# About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)
    
# About Fertilizers
elif(app_mode=="Fertilizers"):
    st.header("Fertilizers")
   
    st.markdown("""
                #### Creating your own fertilizers can be a sustainable and cost-effective way to nourish your plants. Here are some steps to make your own fertilizers:
                  """)    
    
    image_path = "fertilizersfinal.jpeg"
    st.image(image_path,use_column_width=True)
    
    st.markdown("""
1) Understand Plant Needs: Different plants have different nutritional requirements. Before making your own fertilizer, understand the specific needs of the plants you want to fertilize. Common nutrients required by plants include nitrogen (N), phosphorus (P), and potassium (K), as well as micronutrients like calcium, magnesium, and iron.

2) Source Ingredients: Many household items and organic materials can be used to create homemade fertilizers. Some common ingredients include:

-> Compost: Rich in organic matter and nutrients, compost can serve as an excellent fertilizer.

-> Manure: Well-aged animal manure (such as cow, horse, or chicken manure) provides nitrogen, phosphorus, and potassium.

-> Coffee grounds: Coffee grounds add nitrogen to the soil.

-> Eggshells: Crushed eggshells are a source of calcium.

-> Seaweed: Seaweed contains trace minerals and growth hormones beneficial to plants.

-> Banana peels: Rich in potassium, banana peels can be added to fertilizer mixtures, therefore improving the growth of plant.

-> Epsom salt: Provides magnesium, which is essential for plant growth.

-> Fish emulsion: Made from fish waste, this is a rich source of nitrogen and other nutrients.

-> Molasses: Contains sugars and micronutrients, and can stimulate microbial activity in the soil.


3) Choose a Recipe: Based on the needs of your plants and the ingredients you have available, select a recipe for your homemade fertilizer. There are many recipes available online, or you can experiment to find what works best for your plants.

4) Prepare the Fertilizer: Once you've chosen a recipe, mix the ingredients together thoroughly. Depending on the recipe, you may need to blend, shred, or crush the ingredients to create a uniform mixture.

5) Application: Apply the homemade fertilizer to your plants according to the instructions for the specific fertilizer blend. Some fertilizers can be applied directly to the soil, while others may need to be diluted with water before application. Be careful not to over-fertilize, as this can harm your plants.

6) Monitor Plant Health: Keep an eye on your plants after applying homemade fertilizer to ensure they're responding well. Adjust your fertilization practices as needed based on plant health and growth.

7) Experiment and Learn: Homemade fertilizers can be a bit of trial and error. Experiment with different ingredients and ratios to find what works best for your plants. Keep notes on your experiments so you can learn from them over time.

8) Safety Precautions: Some homemade fertilizers, such as those containing manure, may carry pathogens or odors. Handle these materials safely and wash your hands thoroughly after handling them.

Remember, while homemade fertilizers can be effective and environmentally friendly, they may not provide precise nutrient ratios like commercial fertilizers. Regular soil testing can help ensure your plants are getting the nutrients they need.
            

                """)    

# Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    # Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        # Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success("Model is Predicting it is a {}".format(class_name[result_index]))



# Stastics
elif(app_mode=="Stastics"):
    st.header("Stastics")
   
    st.markdown("""
                ####  STASTICS DASHBOARD
                  """)    
    st.markdown("""
                Histogram of Plant Diseases
                """)
    image_path = "dashboard1.jpg"
    st.image(image_path,use_column_width=True)
    
    
    st.markdown("""
                Disease causing Agents
                """)       
    image_path = "dashboard2.png"
    st.image(image_path,use_column_width=True)
    
    
    st.markdown("""
                Accuracy of different Diseases
                """)
    image_path = "dashboard3.jpg"
    st.image(image_path,use_column_width=True)
    
    
    st.markdown("""
                Training and Testing
                """)
    image_path = "dataset6.png"
    st.image(image_path,use_column_width=True)