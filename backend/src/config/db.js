import mongoose from "mongoose";

export async function connectdb(){
await mongoose.connect(process.env.MONGO_URI)
.then(()=>{
    console.log("connected db")
})

}