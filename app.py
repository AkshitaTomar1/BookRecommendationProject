import pickle 
import streamlit as st
import numpy as np

st.header("Books recommendation System using Machine Learning")
model=pickle.load(open('artifacts/model.pkl','rb'))
book_names=pickle.load(open('artifacts/book_names.pkl','rb'))
final_rating=pickle.load(open('artifacts/final_rating.pkl','rb'))
book_pivot=pickle.load(open('artifacts/book_pivot.pkl','rb'))

def fetch_poster(self,suggestion):
        
            book_name = []
            ids_index = []
            poster_url = []
            book_pivot =  pickle.load(open(self.recommendation_config.book_pivot_serialized_objects,'rb'))
            final_rating =  pickle.load(open(self.recommendation_config.final_rating_serialized_objects,'rb'))

            for book_id in suggestion:
                book_name.append(book_pivot.index[book_id])

            for name in book_name[0]: 
                ids = np.where(final_rating['Book-Title'] == name)[0][0]
                ids_index.append(ids)

            for idx in ids_index:
                url = final_rating.iloc[idx]['Image-URL-L']
                poster_url.append(url)

            return poster_url
        
 
 
 
def recommend_book(self,book_name):
        
            books_list = []
            model = pickle.load(open(self.recommendation_config.trained_model_path,'rb'))
            book_pivot =  pickle.load(open(self.recommendation_config.book_pivot_serialized_objects,'rb'))
            book_id = np.where(book_pivot.index == book_name)[0][0]
            distance, suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6 )

            poster_url = self.fetch_poster(suggestion)
            
            for i in range(len(suggestion)):
                    books = book_pivot.index[suggestion[i]]
                    for j in books:
                        books_list.append(j)
            return books_list , poster_url   



selected_books=st.selectbox("Type or select a book",book_names)

if st.button('Show Recommendation'):
       recommendation_books,poster_url=recommend_book(selected_books)
       col1,col2,col3,col4,col5=st.columns(5)

       with col1:
              st.text(recommendation_books[1])
              st.image(poster_url[1])
       with col2:
              st.text(recommendation_books[2])
              st.image(poster_url[2])
                    
       with col3:
              st.text(recommendation_books[3])
              st.image(poster_url[3])
              
       with col4:
              st.text(recommendation_books[4])
              st.image(poster_url[4])

       with col5:
              st.text(recommendation_books[5])
              st.image(poster_url[5])
              
              
