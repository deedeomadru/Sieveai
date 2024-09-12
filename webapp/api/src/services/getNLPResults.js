// import http from '../utils/httpInstance';

// const getNLPResults = async (payload, cb) => {
//   try {
//     const { data } = await http.post('/process', payload);
//     return data;
//   } catch (err) {
//     console.log(err);
//     cb(new Error('error during NLP data proccessing'));
//   }
// };

// export default getNLPResults;


import http from '../utils/httpInstance';

const getNLPResults = async (payload) => {
  console.log('tag before try nlpresults',payload)
  try {
    const { data } = await http.post('/process', payload);
    console.log('tag after try nlpresults')

    return data;
  } catch (err) {
    // Log the full error object to see detailed error information
    console.error('Error during NLP data processing:', err);
    
    // Optionally, rethrow the error if you want to handle it further up the call stack
    throw err;
  }
};

export default getNLPResults;
