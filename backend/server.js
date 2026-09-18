import app from "./src/app.js";
import { config } from "dotenv";
import { connectdb } from "./src/config/db.js";
config();
connectdb()



app.listen(3000,()=>{
    console.log("server is running on 3000")
})
