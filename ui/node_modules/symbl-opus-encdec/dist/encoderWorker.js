"use strict";

var OggOpusEncoder, OpusEncoderLib;
if(typeof require === 'function'){
  OpusEncoderLib = require('https://symbl-sdk-cdn-bucket.storage.googleapis.com/js/ga/symbl-opus-encdec/0.1.2/dist/libopus-encoder.js');
  OggOpusEncoder = require('https://symbl-sdk-cdn-bucket.storage.googleapis.com/js/ga/symbl-opus-encdec/0.1.2/dist/oggOpusEncoder.min.js').OggOpusEncoder;
} else if('function' === typeof importScripts) {
  importScripts('https://symbl-sdk-cdn-bucket.storage.googleapis.com/js/ga/symbl-opus-encdec/0.1.2/dist/libopus-encoder.js');
  importScripts('https://symbl-sdk-cdn-bucket.storage.googleapis.com/js/ga/symbl-opus-encdec/0.1.2/dist/oggOpusEncoder.min.js');
}

// Run in AudioWorkletGlobal scope
if (typeof registerProcessor === 'function') {

  class EncoderWorklet extends AudioWorkletProcessor {

    constructor(){
      super();
      this.continueProcess = true;
      this.port.onmessage = ({ data }) => {
        if (this.encoder) {
          switch( data['command'] ){

            case 'getHeaderPages':
              this.postPage(this.encoder.generateIdPage());
              this.postPage(this.encoder.generateCommentPage());
              break;

            case 'done':
              this.encoder.encodeFinalFrame().forEach(pageData => this.postPage(pageData));
              this.encoder.destroy();
              delete this.encoder;
              this.port.postMessage( {message: 'done'} );
              break;

            case 'flush':
              this.postPage(this.encoder.flush());
              this.port.postMessage( {message: 'flushed'} );
              break;

            default:
              // Ignore any unknown commands and continue recieving commands
          }
        }

        switch( data['command'] ){

          case 'close':
            this.continueProcess = false;
            break;

          case 'init':
            this.encoder = new OggOpusEncoder( data, OpusEncoderLib );
            this.port.postMessage( {message: 'ready'} );
            break;

          default:
            // Ignore any unknown commands and continue recieving commands
        }
      }
    }

    process(inputs) {
      if (this.encoder && inputs[0] && inputs[0].length && inputs[0][0] && inputs[0][0].length){
        this.encoder.encode( inputs[0] ).forEach(pageData => this.postPage(pageData));
      }
      return this.continueProcess;
    }

    postPage(pageData) {
      if (pageData) {
        this.port.postMessage( pageData, [pageData.page.buffer] );
      }
    }
  }

  registerProcessor('encoder-worklet', EncoderWorklet);
}

// run in scriptProcessor worker scope
else {
  var encoder;
  var postPageGlobal = (pageData) => {
    if (pageData) {
      postMessage( pageData, [pageData.page.buffer] );
    }
  }

  onmessage = ({ data }) => {
    if (encoder) {
      switch( data['command'] ){

        case 'encode':
          encoder.encode( data['buffers'] ).forEach(pageData => postPageGlobal(pageData));
          break;

        case 'getHeaderPages':
          postPageGlobal(encoder.generateIdPage());
          postPageGlobal(encoder.generateCommentPage());
          break;

        case 'done':
          encoder.encodeFinalFrame().forEach(pageData => postPageGlobal(pageData));
          encoder.destroy();
          encoder = null;
          postMessage( {message: 'done'} );
          break;

        case 'flush':
          postPageGlobal(encoder.flush());
          postMessage( {message: 'flushed'} );
          break;

        default:
          // Ignore any unknown commands and continue recieving commands
      }
    }

    switch( data['command'] ){

      case 'close':
        close();
        break;

      case 'init':
        encoder = new OggOpusEncoder( data, OpusEncoderLib );
        postMessage( {message: 'ready'} );
        break;

      default:
        // Ignore any unknown commands and continue recieving commands
    }
  };
}


// Exports for unit testing.
var module = module || {};
module.exports = {
  OpusEncoderLib: OpusEncoderLib,
  OggOpusEncoder: OggOpusEncoder
};
